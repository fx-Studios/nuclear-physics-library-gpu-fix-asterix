! models/two_phase_flow.f90
!
! Two-phase flow model for nuclear reactor simulation.
! Implements drift-flux model with:
! - Void fraction correlations (Zuber-Findlay, Chexal-Lellouche)
! - Flow regime maps (bubbly, slug, churn, annular, mist)
! - Critical heat flux correlations (Groeneveld, Bowring, CISE-4)
! - Subcooled boiling onset
! - Steam quality and void tracking
! - Two-phase pressure drop
!
! Usage:
!   use two_phase_flow
!   type(two_phase_config_t) :: config
!   type(two_phase_state_t) :: state
!   
!   call two_phase_init(state, nx, ny, nz, dx, dy, dz, config)
!   call two_phase_set_geometry(state, diameter, heated_perimeter)
!   call two_phase_step(state, T, p, mass_flux, heat_flux, dt)
!   call two_phase_get_void_fraction(state, alpha)
!   call two_phase_destroy(state)
!
module two_phase_flow
    use kinds, only: wp, i32
    use constants, only: PI, TOL_DEFAULT, G_GRAV
    use compute_mode, only: use_gpu, gpu_min_workload
#ifdef COMPUTATION_MODE_GPU
    use two_phase_flow_gpu, only: tp_gpu_state_t, tp_gpu_init, tp_gpu_cleanup, &
                                    tp_gpu_step, tp_gpu_copy_from_device
#endif
#ifdef COMPUTATION_MODE_HYBRID
    use two_phase_flow_gpu, only: tp_gpu_state_t, tp_gpu_init, tp_gpu_cleanup, &
                                    tp_gpu_step, tp_gpu_copy_from_device
#endif
    implicit none
    private
    
    ! Public interface
    public :: two_phase_config_t, two_phase_state_t
    public :: two_phase_init, two_phase_destroy
    public :: two_phase_set_geometry, two_phase_set_conditions
    public :: two_phase_step
    public :: two_phase_get_void_fraction, two_phase_get_quality
    public :: two_phase_get_flow_regime, two_phase_check_chf
    public :: water_properties_t, get_water_properties
    
    ! Flow regime types
    integer, parameter, public :: REGIME_SINGLE_PHASE = 0
    integer, parameter, public :: REGIME_SUBCOOLED_BOILING = 1
    integer, parameter, public :: REGIME_BUBBLY = 2
    integer, parameter, public :: REGIME_SLUG = 3
    integer, parameter, public :: REGIME_CHURN = 4
    integer, parameter, public :: REGIME_ANNULAR = 5
    integer, parameter, public :: REGIME_MIST = 6
    integer, parameter, public :: REGIME_DROPLET = 7
    
    ! Void correlation types
    integer, parameter, public :: VOID_ZUBER_FINDLAY_ID = 1
    integer, parameter, public :: VOID_CHEXAL_LELLOUCHE_ID = 2
    integer, parameter, public :: VOID_DIX_ID = 3
    integer, parameter, public :: VOID_HOMOGENEOUS_ID = 4
    
    ! CHF correlation types
    integer, parameter, public :: CHF_GROENEVELD_ID = 1
    integer, parameter, public :: CHF_BOWRING_ID = 2
    integer, parameter, public :: CHF_CISE4_ID = 3
    integer, parameter, public :: CHF_BIASI_ID = 4
    
    ! Water/steam properties at given T, p
    type :: water_properties_t
        real(wp) :: T           ! Temperature [K]
        real(wp) :: p           ! Pressure [Pa]
        real(wp) :: rho_l       ! Liquid density [kg/m³]
        real(wp) :: rho_g       ! Vapour density [kg/m³]
        real(wp) :: mu_l        ! Liquid viscosity [Pa·s]
        real(wp) :: mu_g        ! Vapour viscosity [Pa·s]
        real(wp) :: k_l         ! Liquid conductivity [W/m·K]
        real(wp) :: k_g         ! Vapour conductivity [W/m·K]
        real(wp) :: cp_l        ! Liquid specific heat [J/kg·K]
        real(wp) :: cp_g        ! Vapour specific heat [J/kg·K]
        real(wp) :: h_l         ! Liquid enthalpy [J/kg]
        real(wp) :: h_g         ! Vapour enthalpy [J/kg]
        real(wp) :: h_fg        ! Latent heat [J/kg]
        real(wp) :: sigma       ! Surface tension [N/m]
        real(wp) :: T_sat       ! Saturation temperature [K]
    end type water_properties_t
    
    ! Configuration
    type :: two_phase_config_t
        integer :: void_correlation = VOID_CHEXAL_LELLOUCHE_ID
        integer :: chf_correlation = CHF_GROENEVELD_ID
        logical :: include_subcooled_boiling = .true.
        logical :: track_flow_regime = .true.
        real(wp) :: onset_boiling_superheat = 5.0_wp  ! K
        real(wp) :: chf_safety_factor = 1.0_wp
        real(wp) :: min_quality = -0.5_wp
        real(wp) :: max_quality = 1.5_wp
    end type two_phase_config_t
    
    ! Two-phase flow state
    type :: two_phase_state_t
        ! Grid dimensions
        integer :: nx, ny, nz
        real(wp) :: dx, dy, dz
        
        ! Geometry (per cell or channel)
        real(wp), allocatable :: diameter(:, :, :)          ! Hydraulic diameter [m]
        real(wp), allocatable :: flow_area(:, :, :)         ! Flow area [m²]
        real(wp), allocatable :: heated_perimeter(:, :, :)  ! Heated perimeter [m]
        real(wp), allocatable :: wetted_perimeter(:, :, :)  ! Wetted perimeter [m]
        
        ! Flow conditions
        real(wp), allocatable :: mass_flux(:, :, :)         ! G [kg/m²·s]
        real(wp), allocatable :: heat_flux(:, :, :)         ! q" [W/m²]
        real(wp), allocatable :: pressure(:, :, :)          ! p [Pa]
        real(wp), allocatable :: temperature(:, :, :)       ! T [K]
        
        ! Two-phase parameters
        real(wp), allocatable :: void_fraction(:, :, :)     ! α [-]
        real(wp), allocatable :: quality(:, :, :)           ! x [-]
        real(wp), allocatable :: slip_ratio(:, :, :)        ! S = v_g/v_l [-]
        real(wp), allocatable :: velocity_liquid(:, :, :)   ! v_l [m/s]
        real(wp), allocatable :: velocity_vapour(:, :, :)   ! v_g [m/s]
        
        ! Regime and safety
        integer, allocatable :: flow_regime(:, :, :)
        real(wp), allocatable :: chf_ratio(:, :, :)         ! DNBR or CHFR
        logical, allocatable :: boiling(:, :, :)
        
        ! Water properties
        type(water_properties_t), allocatable :: props(:, :, :)
        
        ! Configuration
        type(two_phase_config_t) :: config
        
        ! Statistics
        integer :: n_boiling_cells = 0
        integer :: n_chf_violations = 0
        real(wp) :: max_void_fraction = 0.0_wp
        real(wp) :: min_chf_ratio = 1.0e10_wp
    end type two_phase_state_t
    
contains

    !> Initialize two-phase flow state
    subroutine two_phase_init(state, nx, ny, nz, dx, dy, dz, config)
        type(two_phase_state_t), intent(out) :: state
        integer, intent(in) :: nx, ny, nz
        real(wp), intent(in) :: dx, dy, dz
        type(two_phase_config_t), intent(in), optional :: config
        
        state%nx = nx
        state%ny = ny
        state%nz = nz
        state%dx = dx
        state%dy = dy
        state%dz = dz
        
        ! Allocate geometry
        allocate(state%diameter(nx, ny, nz))
        allocate(state%flow_area(nx, ny, nz))
        allocate(state%heated_perimeter(nx, ny, nz))
        allocate(state%wetted_perimeter(nx, ny, nz))
        
        ! Allocate flow conditions
        allocate(state%mass_flux(nx, ny, nz))
        allocate(state%heat_flux(nx, ny, nz))
        allocate(state%pressure(nx, ny, nz))
        allocate(state%temperature(nx, ny, nz))
        
        ! Allocate two-phase parameters
        allocate(state%void_fraction(nx, ny, nz))
        allocate(state%quality(nx, ny, nz))
        allocate(state%slip_ratio(nx, ny, nz))
        allocate(state%velocity_liquid(nx, ny, nz))
        allocate(state%velocity_vapour(nx, ny, nz))
        
        ! Allocate regime tracking
        allocate(state%flow_regime(nx, ny, nz))
        allocate(state%chf_ratio(nx, ny, nz))
        allocate(state%boiling(nx, ny, nz))
        
        ! Allocate properties
        allocate(state%props(nx, ny, nz))
        
        ! Initialize with default values
        state%diameter = 0.01_wp           ! 10 mm default
        state%flow_area = PI * 0.01_wp**2 / 4.0_wp
        state%heated_perimeter = PI * 0.01_wp
        state%wetted_perimeter = PI * 0.01_wp
        
        state%mass_flux = 1000.0_wp        ! 1000 kg/m²·s
        state%heat_flux = 0.0_wp
        state%pressure = 7.0e6_wp          ! 7 MPa (typical BWR)
        state%temperature = 558.0_wp       ! Saturation at 7 MPa
        
        state%void_fraction = 0.0_wp
        state%quality = 0.0_wp
        state%slip_ratio = 1.0_wp
        state%velocity_liquid = 0.0_wp
        state%velocity_vapour = 0.0_wp
        
        state%flow_regime = REGIME_SINGLE_PHASE
        state%chf_ratio = 1.0e10_wp
        state%boiling = .false.

        ! Step-11: initialise water properties at the default (T, p)
        ! so that downstream consumers (e.g. fuel_compute_convection_
        ! coefficients) can be called before the first two_phase_step.
        ! Without this, props%mu_l / props%k_l default to 0 and the
        ! Dittus-Boelter helper degenerates.
        block
            type(water_properties_t) :: w
            w = get_water_properties(state%temperature(1, 1, 1), &
                                     state%pressure(1, 1, 1))
            state%props = w
        end block

        ! Configuration
        if (present(config)) then
            state%config = config
        end if
    end subroutine two_phase_init
    
    !> Destroy two-phase flow state
    subroutine two_phase_destroy(state)
        type(two_phase_state_t), intent(inout) :: state
        
        if (allocated(state%diameter)) deallocate(state%diameter)
        if (allocated(state%flow_area)) deallocate(state%flow_area)
        if (allocated(state%heated_perimeter)) deallocate(state%heated_perimeter)
        if (allocated(state%wetted_perimeter)) deallocate(state%wetted_perimeter)
        if (allocated(state%mass_flux)) deallocate(state%mass_flux)
        if (allocated(state%heat_flux)) deallocate(state%heat_flux)
        if (allocated(state%pressure)) deallocate(state%pressure)
        if (allocated(state%temperature)) deallocate(state%temperature)
        if (allocated(state%void_fraction)) deallocate(state%void_fraction)
        if (allocated(state%quality)) deallocate(state%quality)
        if (allocated(state%slip_ratio)) deallocate(state%slip_ratio)
        if (allocated(state%velocity_liquid)) deallocate(state%velocity_liquid)
        if (allocated(state%velocity_vapour)) deallocate(state%velocity_vapour)
        if (allocated(state%flow_regime)) deallocate(state%flow_regime)
        if (allocated(state%chf_ratio)) deallocate(state%chf_ratio)
        if (allocated(state%boiling)) deallocate(state%boiling)
        if (allocated(state%props)) deallocate(state%props)
    end subroutine two_phase_destroy
    
    !> Set geometry for a region
    subroutine two_phase_set_geometry(state, diameter, heated_perim, &
                                      i1, i2, j1, j2, k1, k2)
        type(two_phase_state_t), intent(inout) :: state
        real(wp), intent(in) :: diameter
        real(wp), intent(in), optional :: heated_perim
        integer, intent(in), optional :: i1, i2, j1, j2, k1, k2
        
        integer :: imin, imax, jmin, jmax, kmin, kmax
        real(wp) :: D_h, A_flow, P_heated, P_wetted
        
        imin = 1; imax = state%nx
        jmin = 1; jmax = state%ny
        kmin = 1; kmax = state%nz
        
        if (present(i1)) imin = i1
        if (present(i2)) imax = i2
        if (present(j1)) jmin = j1
        if (present(j2)) jmax = j2
        if (present(k1)) kmin = k1
        if (present(k2)) kmax = k2
        
        D_h = diameter
        A_flow = PI * D_h**2 / 4.0_wp
        P_wetted = PI * D_h
        
        if (present(heated_perim)) then
            P_heated = heated_perim
        else
            P_heated = P_wetted
        end if
        
        state%diameter(imin:imax, jmin:jmax, kmin:kmax) = D_h
        state%flow_area(imin:imax, jmin:jmax, kmin:kmax) = A_flow
        state%heated_perimeter(imin:imax, jmin:jmax, kmin:kmax) = P_heated
        state%wetted_perimeter(imin:imax, jmin:jmax, kmin:kmax) = P_wetted
    end subroutine two_phase_set_geometry
    
    !> Set flow conditions
    subroutine two_phase_set_conditions(state, T, p, G, q_prime_prime)
        type(two_phase_state_t), intent(inout) :: state
        real(wp), intent(in), optional :: T(:, :, :)
        real(wp), intent(in), optional :: p(:, :, :)
        real(wp), intent(in), optional :: G(:, :, :)
        real(wp), intent(in), optional :: q_prime_prime(:, :, :)
        
        if (present(T)) state%temperature = T
        if (present(p)) state%pressure = p
        if (present(G)) state%mass_flux = G
        if (present(q_prime_prime)) state%heat_flux = q_prime_prime
    end subroutine two_phase_set_conditions
    
    !> Main time step - update two-phase flow parameters
    subroutine two_phase_step(state, T, p, G, q_prime_prime, dt)
        type(two_phase_state_t), intent(inout) :: state
        real(wp), intent(in) :: T(:, :, :)
        real(wp), intent(in) :: p(:, :, :)
        real(wp), intent(in) :: G(:, :, :)
        real(wp), intent(in) :: q_prime_prime(:, :, :)
        real(wp), intent(in) :: dt
        
        integer :: i, j, k
        
#if defined(COMPUTATION_MODE_GPU) || defined(COMPUTATION_MODE_HYBRID)
        ! GPU dispatch: offload per-cell two-phase computation to GPU when
        ! the problem is large enough.
        if (use_gpu()) then
            if (state%nx * state%ny * state%nz >= gpu_min_workload()) then
                call two_phase_step_gpu(state, T, p, G, q_prime_prime, dt)
                return
            end if
        end if
#endif
        
        ! Update conditions
        state%temperature = T
        state%pressure = p
        state%mass_flux = G
        state%heat_flux = q_prime_prime
        
        ! Reset statistics
        state%n_boiling_cells = 0
        state%n_chf_violations = 0
        state%max_void_fraction = 0.0_wp
        state%min_chf_ratio = 1.0e10_wp
        
        ! Update each cell.
        ! OpenMP: each cell is independent except for four shared statistics
        ! (n_boiling_cells, n_chf_violations, max_void_fraction, min_chf_ratio)
        ! which are protected by !$OMP ATOMIC inside update_cell / check_chf.
        !$OMP PARALLEL DO PRIVATE(i, j, k) SCHEDULE(STATIC)
        do k = 1, state%nz
            do j = 1, state%ny
                do i = 1, state%nx
                    call update_cell(state, i, j, k)
                end do
            end do
        end do
        !$OMP END PARALLEL DO
    end subroutine two_phase_step
    
    !> Update single cell two-phase parameters
    subroutine update_cell(state, i, j, k)
        type(two_phase_state_t), intent(inout) :: state
        integer, intent(in) :: i, j, k
        
        real(wp) :: T, p, G, q_pp, D_h
        real(wp) :: x, alpha, S, v_l, v_g
        type(water_properties_t) :: props
        
        T = state%temperature(i, j, k)
        p = state%pressure(i, j, k)
        G = state%mass_flux(i, j, k)
        q_pp = state%heat_flux(i, j, k)
        D_h = state%diameter(i, j, k)
        
        ! Get water/steam properties
        props = get_water_properties(T, p)
        state%props(i, j, k) = props
        
        ! Compute quality
        x = compute_quality(T, p, q_pp, G, D_h, props, &
                           state%heated_perimeter(i, j, k), &
                           state%config%include_subcooled_boiling)
        
        ! Clamp quality
        x = max(state%config%min_quality, min(state%config%max_quality, x))
        state%quality(i, j, k) = x
        
        ! Check if boiling
        if (x > -TOL_DEFAULT) then
            state%boiling(i, j, k) = .true.
            !$OMP ATOMIC
            state%n_boiling_cells = state%n_boiling_cells + 1
        else
            state%boiling(i, j, k) = .false.
        end if
        
        ! Compute void fraction
        if (x <= 0.0_wp) then
            ! Subcooled or single-phase liquid
            if (state%config%include_subcooled_boiling .and. &
                T > props%T_sat - state%config%onset_boiling_superheat) then
                ! Subcooled boiling - use correlation
                alpha = compute_subcooled_void(T, p, G, q_pp, props)
            else
                alpha = 0.0_wp
            end if
        else
            ! Two-phase or superheated
            alpha = compute_void_fraction(x, G, p, D_h, props, &
                                         state%config%void_correlation)
        end if
        
        state%void_fraction(i, j, k) = alpha
        !$OMP ATOMIC UPDATE
        state%max_void_fraction = max(state%max_void_fraction, alpha)
        
        ! Compute slip ratio and velocities
        if (alpha > TOL_DEFAULT .and. alpha < 1.0_wp - TOL_DEFAULT) then
            S = compute_slip_ratio(x, alpha, props)
            v_l = G * (1.0_wp - x) / (props%rho_l * (1.0_wp - alpha))
            v_g = S * v_l
        else if (alpha <= TOL_DEFAULT) then
            S = 1.0_wp
            v_l = G / props%rho_l
            v_g = 0.0_wp
        else
            S = 1.0_wp
            v_l = 0.0_wp
            v_g = G * x / props%rho_g
        end if
        
        state%slip_ratio(i, j, k) = S
        state%velocity_liquid(i, j, k) = v_l
        state%velocity_vapour(i, j, k) = v_g
        
        ! Determine flow regime
        if (state%config%track_flow_regime) then
            state%flow_regime(i, j, k) = determine_flow_regime(x, alpha, G, D_h, props)
        end if
        
        ! Check CHF
        if (q_pp > TOL_DEFAULT .and. state%boiling(i, j, k)) then
            call check_chf_condition(state, i, j, k, props)
        else
            state%chf_ratio(i, j, k) = 1.0e10_wp
        end if
    end subroutine update_cell
    
    !> Compute thermodynamic quality
    function compute_quality(T, p, q_pp, G, D_h, props, P_heated, subcooled) result(x)
        real(wp), intent(in) :: T, p, q_pp, G, D_h, P_heated
        type(water_properties_t), intent(in) :: props
        logical, intent(in) :: subcooled
        real(wp) :: x
        
        real(wp) :: h, h_inlet, delta_h, z_boiling, A_flow
        
        ! Enthalpy from temperature
        if (T < props%T_sat) then
            h = props%h_l + props%cp_l * (T - props%T_sat)
        else
            ! Simple interpolation between h_l and h_g
            h = props%h_l + (T - props%T_sat) / 50.0_wp * props%h_fg
            h = min(h, props%h_g)
        end if
        
        ! Quality from enthalpy
        if (h < props%h_l) then
            x = (h - props%h_l) / props%h_fg  ! Negative (subcooled)
        else if (h > props%h_g) then
            x = 1.0_wp + (h - props%h_g) / props%h_fg  ! Superheated
        else
            x = (h - props%h_l) / props%h_fg
        end if
        
        ! Alternative: compute from heat balance
        if (q_pp > TOL_DEFAULT .and. G > TOL_DEFAULT) then
            ! Assume equilibrium quality from heat input
            ! x = (q"·P_h·z) / (G·A·h_fg)
            ! For single cell, use simple model
            A_flow = PI * D_h**2 / 4.0_wp
            delta_h = q_pp * P_heated / (G * A_flow)
            
            ! Use enthalpy-based quality as primary, heat balance as check
            ! (In full channel model, integrate delta_h along flow path)
        end if
    end function compute_quality
    
    !> Compute void fraction using selected correlation
    function compute_void_fraction(x, G, p, D_h, props, correlation) result(alpha)
        real(wp), intent(in) :: x, G, p, D_h
        type(water_properties_t), intent(in) :: props
        integer, intent(in) :: correlation
        real(wp) :: alpha
        
        select case(correlation)
        case(VOID_ZUBER_FINDLAY_ID)
            alpha = void_zuber_findlay(x, G, props)
        case(VOID_CHEXAL_LELLOUCHE_ID)
            alpha = void_chexal_lellouche(x, G, p, D_h, props)
        case(VOID_DIX_ID)
            alpha = void_dix(x, G, props)
        case(VOID_HOMOGENEOUS_ID)
            alpha = void_homogeneous(x, props)
        case default
            alpha = void_chexal_lellouche(x, G, p, D_h, props)
        end select
        
        ! Ensure physical bounds
        alpha = max(0.0_wp, min(1.0_wp, alpha))
    end function compute_void_fraction
    
    !> Zuber-Findlay drift-flux correlation
    function void_zuber_findlay(x, G, props) result(alpha)
        real(wp), intent(in) :: x, G
        type(water_properties_t), intent(in) :: props
        real(wp) :: alpha
        
        real(wp) :: C0, v_gj, j, beta
        
        ! Distribution parameter (typically 1.0-1.2 for vertical flow)
        C0 = 1.13_wp
        
        ! Drift velocity (Zuber-Findlay, m/s)
        v_gj = 1.41_wp * ((props%sigma * G_GRAV * (props%rho_l - props%rho_g)) / &
                          props%rho_l**2)**0.25_wp
        
        ! Mixture velocity
        j = G * (x / props%rho_g + (1.0_wp - x) / props%rho_l)
        
        ! Volumetric quality
        beta = x * props%rho_l / (x * props%rho_l + (1.0_wp - x) * props%rho_g)
        
        ! Drift-flux relation
        alpha = beta / (C0 + v_gj / j)
        
        ! Handle limiting cases
        if (j < TOL_DEFAULT) alpha = beta
    end function void_zuber_findlay
    
    !> Chexal-Lellouche correlation (EPRI, widely used)
    function void_chexal_lellouche(x, G, p, D_h, props) result(alpha)
        real(wp), intent(in) :: x, G, p, D_h
        type(water_properties_t), intent(in) :: props
        real(wp) :: alpha
        
        real(wp) :: K, C0, v_gj, j_g, j_l, A, B, C, D
        real(wp) :: rho_ratio, mu_ratio, Re_l, We_l
        real(wp) :: p_crit, p_star, j_star
        
        ! Fluid property ratios
        rho_ratio = props%rho_g / props%rho_l
        mu_ratio = props%mu_g / props%mu_l
        
        ! Dimensionless numbers
        j_l = G * (1.0_wp - x) / props%rho_l
        Re_l = G * (1.0_wp - x) * D_h / props%mu_l
        We_l = G**2 * D_h / (props%rho_l * props%sigma)
        
        ! Dimensionless pressure (critical pressure of water: 22.064 MPa)
        p_crit = 22.064e6_wp
        p_star = p / p_crit
        
        ! Distribution parameter
        A = rho_ratio**0.1_wp
        C0 = (1.0_wp - x)**A * (1.0_wp + x * (A / (1.0_wp - A)))
        C0 = max(1.0_wp, min(1.5_wp, C0))
        
        ! Drift velocity components
        B = (1.0_wp - p_star)**0.5_wp
        C = (1.0_wp - x)**0.1_wp
        D = (rho_ratio)**(-0.5_wp)
        
        v_gj = 2.9_wp * B * C * D * ((props%sigma * G_GRAV * &
               (props%rho_l - props%rho_g)) / props%rho_l**2)**0.25_wp
        
        ! Mixture volumetric flux
        j_g = G * x / props%rho_g
        j_star = j_g + j_l
        
        ! Void fraction
        if (j_star > TOL_DEFAULT) then
            alpha = j_g / (C0 * j_star + v_gj)
        else
            alpha = 0.0_wp
        end if
        
        ! Corrections for low flow or high void
        K = 1.0_wp
        if (alpha > 0.8_wp) then
            K = (1.0_wp - alpha)**2
            alpha = alpha * K
        end if
    end function void_chexal_lellouche
    
    !> Dix correlation (simpler alternative)
    function void_dix(x, G, props) result(alpha)
        real(wp), intent(in) :: x, G
        type(water_properties_t), intent(in) :: props
        real(wp) :: alpha
        
        real(wp) :: S, beta
        
        ! Slip ratio correlation
        S = ((props%rho_l / props%rho_g) * ((1.0_wp - x) / x))**0.1_wp
        S = max(1.0_wp, min(S, 3.0_wp))
        
        ! Volumetric quality
        beta = 1.0_wp / (1.0_wp + (1.0_wp - x) / x * props%rho_g / props%rho_l)
        
        ! Void from slip
        alpha = beta / (1.0_wp + beta * (S - 1.0_wp))
    end function void_dix
    
    !> Homogeneous equilibrium model (no slip)
    function void_homogeneous(x, props) result(alpha)
        real(wp), intent(in) :: x
        type(water_properties_t), intent(in) :: props
        real(wp) :: alpha
        
        ! alpha = x·ρ_l / (x·ρ_l + (1-x)·ρ_g)
        alpha = 1.0_wp / (1.0_wp + (1.0_wp - x) / x * props%rho_g / props%rho_l)
    end function void_homogeneous
    
    !> Subcooled boiling void fraction (Levy model)
    function compute_subcooled_void(T, p, G, q_pp, props) result(alpha)
        real(wp), intent(in) :: T, p, G, q_pp
        type(water_properties_t), intent(in) :: props
        real(wp) :: alpha
        
        real(wp) :: delta_T_sub, x_eq, N_sub, alpha_eq
        
        ! Subcooling
        delta_T_sub = props%T_sat - T
        
        ! Equilibrium quality (negative for subcooled)
        x_eq = -delta_T_sub * props%cp_l / props%h_fg
        
        ! Subcooling number
        N_sub = props%cp_l * delta_T_sub / props%h_fg
        
        ! Levy model: α = α_eq * exp(-N_sub)
        alpha_eq = compute_void_fraction(max(0.0_wp, x_eq), G, p, 0.01_wp, props, VOID_HOMOGENEOUS_ID)
        alpha = alpha_eq * exp(-N_sub)
        
        ! Alternative: Saha-Zuber onset criteria
        ! Pe = G·D_h·cp_l / k_l
        ! If Pe < 70000: q_ONB based on heat flux
        ! If Pe > 70000: q_ONB based on subcooling
        
        alpha = max(0.0_wp, min(0.3_wp, alpha))  ! Limit subcooled void
    end function compute_subcooled_void
    
    !> Compute slip ratio from void and quality
    function compute_slip_ratio(x, alpha, props) result(S)
        real(wp), intent(in) :: x, alpha
        type(water_properties_t), intent(in) :: props
        real(wp) :: S
        
        real(wp) :: beta
        
        if (alpha < TOL_DEFAULT .or. x < TOL_DEFAULT) then
            S = 1.0_wp
            return
        end if
        
        ! Volumetric quality
        beta = 1.0_wp / (1.0_wp + (1.0_wp - x) / x * props%rho_g / props%rho_l)
        
        ! S = (β/α) · ((1-α)/(1-β))
        if (abs(1.0_wp - beta) > TOL_DEFAULT .and. abs(1.0_wp - alpha) > TOL_DEFAULT) then
            S = (beta / alpha) * ((1.0_wp - alpha) / (1.0_wp - beta))
            S = max(1.0_wp, min(3.0_wp, S))
        else
            S = 1.0_wp
        end if
    end function compute_slip_ratio
    
    !> Determine flow regime from Baker flow map
    function determine_flow_regime(x, alpha, G, D_h, props) result(regime)
        real(wp), intent(in) :: x, alpha, G, D_h
        type(water_properties_t), intent(in) :: props
        integer :: regime
        
        real(wp) :: j_g, j_l, We_g, Re_l, Fr_g
        
        ! Check for single-phase first
        if (alpha < 0.01_wp) then
            regime = REGIME_SINGLE_PHASE
            return
        else if (alpha > 0.99_wp) then
            regime = REGIME_MIST
            return
        end if
        
        ! Superficial velocities
        j_g = G * x / props%rho_g
        j_l = G * (1.0_wp - x) / props%rho_l
        
        ! Dimensionless numbers
        We_g = props%rho_g * j_g**2 * D_h / props%sigma
        Re_l = props%rho_l * j_l * D_h / props%mu_l
        Fr_g = j_g / sqrt(G_GRAV * D_h)
        
        ! Simplified regime map (Taitel-Dukler for vertical flow)
        if (alpha < 0.25_wp) then
            regime = REGIME_BUBBLY
        else if (alpha < 0.52_wp) then
            if (j_g < 3.0_wp) then
                regime = REGIME_SLUG
            else
                regime = REGIME_CHURN
            end if
        else if (alpha < 0.75_wp) then
            regime = REGIME_CHURN
        else
            if (We_g > 20.0_wp) then
                regime = REGIME_ANNULAR
            else
                regime = REGIME_MIST
            end if
        end if
        
        ! Subcooled boiling check
        if (x < 0.0_wp .and. alpha > 0.01_wp) then
            regime = REGIME_SUBCOOLED_BOILING
        end if
    end function determine_flow_regime
    
    !> Check CHF condition and compute DNBR/CHFR
    subroutine check_chf_condition(state, i, j, k, props)
        type(two_phase_state_t), intent(inout) :: state
        integer, intent(in) :: i, j, k
        type(water_properties_t), intent(in) :: props
        
        real(wp) :: q_chf, q_actual, chfr, G, p, D_h, x
        
        q_actual = state%heat_flux(i, j, k)
        G = state%mass_flux(i, j, k)
        p = state%pressure(i, j, k)
        D_h = state%diameter(i, j, k)
        x = state%quality(i, j, k)
        
        ! Compute critical heat flux
        select case(state%config%chf_correlation)
        case(CHF_GROENEVELD_ID)
            q_chf = chf_groeneveld_lut(G, p, x, D_h, props)
        case(CHF_BOWRING_ID)
            q_chf = chf_bowring(G, p, x, D_h, props)
        case(CHF_CISE4_ID)
            q_chf = chf_cise4(G, p, x, D_h, props)
        case(CHF_BIASI_ID)
            q_chf = chf_biasi(G, p, x, D_h, props)
        case default
            q_chf = chf_groeneveld_lut(G, p, x, D_h, props)
        end select
        
        ! Apply safety factor
        q_chf = q_chf * state%config%chf_safety_factor
        
        ! Compute CHF ratio
        if (q_actual > TOL_DEFAULT) then
            chfr = q_chf / q_actual
        else
            chfr = 1.0e10_wp
        end if
        
        state%chf_ratio(i, j, k) = chfr
        !$OMP ATOMIC UPDATE
        state%min_chf_ratio = min(state%min_chf_ratio, chfr)
        
        ! Check for CHF violation
        if (chfr < 1.0_wp) then
            !$OMP ATOMIC
            state%n_chf_violations = state%n_chf_violations + 1
        end if
    end subroutine check_chf_condition
    
    !> Groeneveld 2006 CHF look-up table (simplified)
    function chf_groeneveld_lut(G, p, x, D_h, props) result(q_chf)
        real(wp), intent(in) :: G, p, x, D_h
        type(water_properties_t), intent(in) :: props
        real(wp) :: q_chf
        
        real(wp) :: p_crit, p_star, G_star, x_star, Y
        
        ! Dimensionless parameters
        p_crit = 22.064e6_wp
        p_star = p / p_crit
        G_star = G / 1000.0_wp  ! Normalize to kg/m²·s
        x_star = max(0.0_wp, min(1.0_wp, x))
        
        ! Simplified correlation (actual LUT has 8000+ points)
        ! This is a rough approximation
        Y = 0.124_wp * G_star**0.68_wp * (1.0_wp - x_star)**0.25_wp
        Y = Y * (1.0_wp - 0.9_wp * p_star)
        
        q_chf = Y * 1.0e6_wp  ! Convert to W/m²
        
        ! Corrections for geometry
        q_chf = q_chf * (0.008_wp / D_h)**0.1_wp
    end function chf_groeneveld_lut
    
    !> Bowring correlation (simple, widely used)
    function chf_bowring(G, p, x, D_h, props) result(q_chf)
        real(wp), intent(in) :: G, p, x, D_h
        type(water_properties_t), intent(in) :: props
        real(wp) :: q_chf
        
        real(wp) :: A, B, C, D_e, F1, F2, F3, F4, p_bar
        
        ! Pressure in bar
        p_bar = p / 1.0e5_wp
        
        ! Bowring constants
        A = 2.317_wp * props%h_fg * G / 1000.0_wp
        B = 0.077_wp * (G / 1000.0_wp)**0.8_wp
        C = 0.347_wp * (G / 1000.0_wp)**0.8_wp
        
        ! Equivalent diameter factor
        D_e = D_h * 1000.0_wp  ! Convert to mm
        
        ! Correction factors
        F1 = (p_bar / 69.0_wp)**0.368_wp
        F2 = 1.0_wp
        F3 = 1.0_wp
        F4 = (0.008_wp / D_h)**0.1_wp
        
        ! CHF
        q_chf = (A - B * x * props%h_fg) / (C + 1.0_wp)
        q_chf = q_chf * F1 * F2 * F3 * F4
        
        q_chf = max(q_chf, 0.0_wp)
    end function chf_bowring
    
    !> CISE-4 correlation (good for subcooled/low quality)
    function chf_cise4(G, p, x, D_h, props) result(q_chf)
        real(wp), intent(in) :: G, p, x, D_h
        type(water_properties_t), intent(in) :: props
        real(wp) :: q_chf
        
        real(wp) :: G_crit, x_crit, p_bar, beta
        
        p_bar = p / 1.0e5_wp
        
        ! Critical mass flux
        G_crit = 3000.0_wp * exp(-0.01_wp * p_bar)
        
        ! Critical quality
        x_crit = -0.1_wp + 0.0012_wp * p_bar
        
        ! Parameter
        beta = 0.4_wp
        
        ! Simplified CISE-4
        if (x < x_crit) then
            q_chf = G * props%h_fg * (x_crit - x) / (1.0_wp + beta)
        else
            q_chf = G * props%h_fg * (1.0_wp - x) / (1.0_wp + beta)
        end if
        
        ! Geometry correction
        q_chf = q_chf * (0.008_wp / D_h)**0.15_wp
    end function chf_cise4
    
    !> Biasi correlation (European, good for high pressure)
    function chf_biasi(G, p, x, D_h, props) result(q_chf)
        real(wp), intent(in) :: G, p, x, D_h
        type(water_properties_t), intent(in) :: props
        real(wp) :: q_chf
        
        real(wp) :: p_bar, a, b, c, d, n, f, G_star
        
        p_bar = p / 1.0e5_wp
        G_star = G / 1000.0_wp
        
        ! Correlation constants
        a = 3780.0_wp
        b = 1.0_wp
        n = -0.4_wp
        
        ! Biasi low quality correlation
        if (x < 0.0_wp) then
            f = 0.7249_wp + 0.099_wp * p_bar * exp(-0.032_wp * p_bar)
            c = -1.159_wp + 0.149_wp * p_bar * exp(-0.019_wp * p_bar)
            d = 0.0_wp
            
            q_chf = a * f * (G_star / 1000.0_wp)**b * (D_h * 1000.0_wp)**c
        else
            f = 0.7249_wp + 0.099_wp * p_bar * exp(-0.032_wp * p_bar)
            q_chf = a * f * (1.0_wp - x) * (G_star / 1000.0_wp)**b
        end if
        
        q_chf = max(q_chf, 0.0_wp) * 1000.0_wp  ! Convert to W/m²
    end function chf_biasi
    
    !> Get water properties at given T, p (simplified IAPWS-IF97)
    function get_water_properties(T, p) result(props)
        real(wp), intent(in) :: T, p
        type(water_properties_t) :: props
        
        real(wp) :: p_sat, T_sat, p_crit, T_crit
        
        props%T = T
        props%p = p
        
        ! Critical point
        T_crit = 647.096_wp  ! K
        p_crit = 22.064e6_wp  ! Pa
        
        ! Saturation temperature (simplified Antoine equation)
        ! ln(p_sat/Pa) ≈ A - B/(T + C)
        if (p < p_crit) then
            T_sat = 373.15_wp + 100.0_wp * log(p / 101325.0_wp) / log(10.0_wp)
            T_sat = max(273.15_wp, min(T_crit, T_sat))
        else
            T_sat = T_crit
        end if
        
        props%T_sat = T_sat
        
        ! Liquid properties (approximate)
        if (T < T_sat) then
            props%rho_l = 1000.0_wp * (1.0_wp - 0.0002_wp * (T - 300.0_wp))
            props%mu_l = 0.001_wp * exp(-0.01_wp * (T - 273.15_wp))
            props%k_l = 0.6_wp
            props%cp_l = 4180.0_wp
            props%h_l = props%cp_l * (T - 273.15_wp)
        else
            props%rho_l = 1000.0_wp * (1.0_wp - 0.001_wp * (T_sat - 300.0_wp))
            props%mu_l = 0.0001_wp
            props%k_l = 0.6_wp
            props%cp_l = 4180.0_wp
            props%h_l = props%cp_l * (T_sat - 273.15_wp)
        end if
        
        ! Vapour properties (ideal gas approximation)
        props%rho_g = p / (461.5_wp * T)  ! R_specific for steam
        props%mu_g = 1.0e-5_wp * (T / 273.15_wp)**0.7_wp
        props%k_g = 0.025_wp * (T / 273.15_wp)**0.8_wp
        props%cp_g = 2000.0_wp
        
        ! Latent heat (Clausius-Clapeyron)
        props%h_fg = 2.5e6_wp * (1.0_wp - (T_sat - 273.15_wp) / 374.0_wp)**0.38_wp
        props%h_g = props%h_l + props%h_fg
        
        ! Surface tension (approximate)
        props%sigma = 0.0589_wp * (1.0_wp - T_sat / T_crit)**1.26_wp
        props%sigma = max(0.001_wp, props%sigma)
    end function get_water_properties
    
    !> Get void fraction field
    subroutine two_phase_get_void_fraction(state, alpha)
        type(two_phase_state_t), intent(in) :: state
        real(wp), intent(out) :: alpha(:, :, :)
        
        alpha = state%void_fraction
    end subroutine two_phase_get_void_fraction
    
    !> Get quality field
    subroutine two_phase_get_quality(state, x)
        type(two_phase_state_t), intent(in) :: state
        real(wp), intent(out) :: x(:, :, :)
        
        x = state%quality
    end subroutine two_phase_get_quality
    
    !> Get flow regime field
    subroutine two_phase_get_flow_regime(state, regime)
        type(two_phase_state_t), intent(in) :: state
        integer, intent(out) :: regime(:, :, :)
        
        regime = state%flow_regime
    end subroutine two_phase_get_flow_regime
    
    !> Check if CHF has occurred
    function two_phase_check_chf(state) result(chf_occurred)
        type(two_phase_state_t), intent(in) :: state
        logical :: chf_occurred
        
        chf_occurred = (state%n_chf_violations > 0)
    end function two_phase_check_chf

#if defined(COMPUTATION_MODE_GPU) || defined(COMPUTATION_MODE_HYBRID)
    ! -----------------------------------------------------------------
    ! GPU dispatch wrapper for the two-phase flow step.
    ! Creates a temporary GPU state, runs the per-cell kernel, and
    ! copies the results back. Statistics are recomputed on the host
    ! from the GPU output arrays.
    ! -----------------------------------------------------------------
    subroutine two_phase_step_gpu(state, T, p, G, q_pp, dt)
        type(two_phase_state_t), intent(inout) :: state
        real(wp), intent(in) :: T(state%nx, state%ny, state%nz)
        real(wp), intent(in) :: p(state%nx, state%ny, state%nz)
        real(wp), intent(in) :: G(state%nx, state%ny, state%nz)
        real(wp), intent(in) :: q_pp(state%nx, state%ny, state%nz)
        real(wp), intent(in) :: dt
        type(tp_gpu_state_t) :: gs
        integer :: i, j, k
        integer :: boiling_int(state%nx, state%ny, state%nz)

        ! Update input conditions on host state
        state%temperature = T
        state%pressure = p
        state%mass_flux = G
        state%heat_flux = q_pp

        ! Initialise and run GPU kernel
        call tp_gpu_init(gs, state%nx, state%ny, state%nz)
        call tp_gpu_step(gs, T, p, G, q_pp, state%diameter, state%heated_perimeter)

        ! Copy results back
        call tp_gpu_copy_from_device(gs, state%quality, state%void_fraction, &
            state%slip_ratio, state%velocity_liquid, state%velocity_vapour, &
            state%chf_ratio, boiling_int)

        ! Recompute statistics on host from GPU output
        state%n_boiling_cells = 0
        state%n_chf_violations = 0
        state%max_void_fraction = 0.0_wp
        state%min_chf_ratio = 1.0e10_wp
        do k = 1, state%nz; do j = 1, state%ny; do i = 1, state%nx
            if (boiling_int(i,j,k) == 1) then
                state%boiling(i,j,k) = .true.
                state%n_boiling_cells = state%n_boiling_cells + 1
            else
                state%boiling(i,j,k) = .false.
            end if
            state%max_void_fraction = max(state%max_void_fraction, state%void_fraction(i,j,k))
            state%min_chf_ratio = min(state%min_chf_ratio, state%chf_ratio(i,j,k))
            if (state%chf_ratio(i,j,k) < 1.0_wp) then
                state%n_chf_violations = state%n_chf_violations + 1
            end if
        end do; end do; end do

        call tp_gpu_cleanup(gs)
    end subroutine two_phase_step_gpu
#endif

end module two_phase_flow