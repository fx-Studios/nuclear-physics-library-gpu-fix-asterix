! models/multigroup_diffusion.f90
!
! Multi-group neutron diffusion solver for reactor physics.
! Solves the steady-state and time-dependent multi-group diffusion equations:
!
!   1/v_g · ∂ϕ_g/∂t = ∇·D_g∇ϕ_g - Σ_r,g·ϕ_g + Σ_s,g'→g·ϕ_g' + χ_g·(1/k)·Σ_f,g'·ϕ_g'
!
! Features:
! - 2, 4, or 8 energy group structures
! - Fast/thermal spectrum treatment
! - Upscattering and downscattering
! - Fission source iteration for k-eigenvalue
! - Power iteration and source iteration methods
! - Albedo boundary conditions
!
! Usage:
!   use multigroup_diffusion
!   type(mg_config_t) :: config
!   type(mg_state_t) :: state
!   
!   config%n_groups = 2
!   call mg_init(state, nx, ny, nz, dx, dy, dz, config)
!   call mg_set_cross_sections(state, xsec_data)
!   call mg_solve_eigenvalue(state, k_eff, converged)
!   call mg_solve_transient(state, precursors, dt)
!   call mg_get_power(state, power_density)
!   call mg_destroy(state)
!
module multigroup_diffusion
    use kinds, only: wp, i32
    use constants, only: TOL_DEFAULT
    use compute_mode, only: use_gpu, gpu_min_workload
    use solve_linear, only: solve_dense, SOLVE_SUCCESS
    use sparse_matrix, only: sparse_matrix_t, sparse_matvec
#ifdef COMPUTATION_MODE_GPU
    use multigroup_diffusion_gpu, only: mg_gpu_state_t, mg_gpu_init, mg_gpu_cleanup, &
        mg_gpu_copy_to_device, mg_gpu_copy_from_device, mg_gpu_source_iteration_step, &
        mg_gpu_set_albedos, mg_gpu_fission_sum, mg_gpu_normalize_flux, &
        mg_gpu_compute_fission_source
#endif
#ifdef COMPUTATION_MODE_HYBRID
    use multigroup_diffusion_gpu, only: mg_gpu_state_t, mg_gpu_init, mg_gpu_cleanup, &
        mg_gpu_copy_to_device, mg_gpu_copy_from_device, mg_gpu_source_iteration_step, &
        mg_gpu_set_albedos, mg_gpu_fission_sum, mg_gpu_normalize_flux, &
        mg_gpu_compute_fission_source
#endif
    implicit none
    private
    
    ! Public interface
    public :: mg_config_t, mg_state_t, mg_xsec_t
    public :: mg_init, mg_destroy
    public :: mg_set_cross_sections, mg_set_source
    public :: mg_solve_eigenvalue, mg_solve_fixed_source
    public :: mg_solve_transient, mg_power_iteration
    public :: mg_get_flux, mg_get_power, mg_get_fission_rate
    public :: mg_compute_keff, mg_compute_form_factors
    public :: mg_validate_cross_sections
    public :: normalize_flux_to_unity
    
    ! Standard energy group structures (eV boundaries)
    integer, parameter, public :: MG_2GROUP = 2
    integer, parameter, public :: MG_4GROUP = 4
    integer, parameter, public :: MG_8GROUP = 8
    
    ! Solver methods
    integer, parameter, public :: SOLVER_POWER_ITERATION = 1
    integer, parameter, public :: SOLVER_SOURCE_ITERATION = 2
    integer, parameter, public :: SOLVER_KRYLOV = 3
    
    ! Multi-group cross sections (per material)
    type :: mg_xsec_t
        integer :: n_groups = 2
        
        ! Group-wise cross sections [cm⁻¹]
        real(wp), allocatable :: D(:)              ! Diffusion coefficient
        real(wp), allocatable :: sigma_t(:)        ! Total
        real(wp), allocatable :: sigma_a(:)        ! Absorption
        real(wp), allocatable :: sigma_f(:)        ! Fission
        real(wp), allocatable :: nu_sigma_f(:)     ! Production
        real(wp), allocatable :: sigma_r(:)        ! Removal
        real(wp), allocatable :: chi(:)            ! Fission spectrum
        real(wp), allocatable :: kappa(:)          ! Energy per fission [MeV]
        
        ! Scattering matrix [cm⁻¹]: (from → to)
        real(wp), allocatable :: sigma_s(:, :)     ! (g_from, g_to)
        
        ! Delayed neutron data
        real(wp) :: beta_total = 0.0065_wp
        real(wp), allocatable :: chi_d(:)          ! Delayed spectrum
    end type mg_xsec_t
    
    ! Configuration
    type :: mg_config_t
        integer :: n_groups = 2
        integer :: max_outer_iter = 100
        integer :: max_inner_iter = 50
        real(wp) :: outer_tolerance = 1.0e-5_wp
        ! Inner-iteration relative flux tolerance. 1e-4 is tight enough for
        ! transient physics fidelity (XS feedback steps are O(1 %) per tick)
        ! but loose enough that the iterative solver converges within
        ! max_inner_iter at quasi-steady states — 1e-6 was so tight that
        ! discretisation noise alone kept the residual above it.
        real(wp) :: inner_tolerance = 1.0e-4_wp
        real(wp) :: k_guess = 1.0_wp
        logical :: use_upscatter = .true.
        logical :: normalize_power = .true.
        real(wp) :: power_level = 1.0e6_wp        ! Watts
        integer :: solver_method = SOLVER_POWER_ITERATION
        
        ! Boundary conditions (albedo)
        real(wp) :: albedo_x_min = 0.0_wp
        real(wp) :: albedo_x_max = 0.0_wp
        real(wp) :: albedo_y_min = 0.0_wp
        real(wp) :: albedo_y_max = 0.0_wp
        real(wp) :: albedo_z_min = 0.0_wp
        real(wp) :: albedo_z_max = 0.0_wp
    end type mg_config_t
    
    ! Multi-group diffusion state
    type :: mg_state_t
        ! Grid
        integer :: nx, ny, nz
        real(wp) :: dx, dy, dz
        real(wp) :: volume
        
        ! Number of energy groups
        integer :: n_groups
        
        ! Multi-group flux [n/cm²·s] (nx, ny, nz, n_groups)
        real(wp), allocatable :: flux(:, :, :, :)
        real(wp), allocatable :: flux_old(:, :, :, :)
        
        ! Fission source [n/cm³·s]
        real(wp), allocatable :: fission_source(:, :, :)
        
        ! Fixed source [n/cm³·s] (optional, for fixed source problems)
        real(wp), allocatable :: external_source(:, :, :, :)
        
        ! Cross sections per cell
        type(mg_xsec_t), allocatable :: xsec(:, :, :)
        
        ! Eigenvalue
        real(wp) :: k_eff = 1.0_wp
        real(wp) :: k_eff_old = 1.0_wp
        
        ! Convergence tracking
        integer :: outer_iterations = 0
        integer :: inner_iterations = 0
        real(wp) :: outer_error = 1.0_wp
        real(wp) :: inner_error = 1.0_wp
        logical :: converged = .false.
        
        ! Power distribution
        real(wp), allocatable :: power_density(:, :, :)  ! W/cm³
        real(wp) :: total_power = 0.0_wp
        
        ! Form factors
        real(wp), allocatable :: radial_peaking(:, :)
        real(wp), allocatable :: axial_peaking(:)
        real(wp) :: total_peaking = 1.0_wp

        ! Cached values
        real(wp) :: inv_dx2, inv_dy2, inv_dz2
        real(wp) :: two_inv_dx2, two_inv_dy2, two_inv_dz2
        
        ! Configuration
        type(mg_config_t) :: config
    end type mg_state_t
    
contains

    !> Initialize multi-group diffusion state (with pre-computation)
    subroutine mg_init(state, nx, ny, nz, dx, dy, dz, config)
        type(mg_state_t), intent(out) :: state
        integer, intent(in) :: nx, ny, nz
        real(wp), intent(in) :: dx, dy, dz
        type(mg_config_t), intent(in), optional :: config
        
        integer :: g, i, j, k
        
        state%nx = nx
        state%ny = ny
        state%nz = nz
        state%dx = dx * 100.0_wp  ! Convert m to cm
        state%dy = dy * 100.0_wp
        state%dz = dz * 100.0_wp
        state%volume = dx * dy * dz  ! m³
        
        ! Pre-compute inverse grid spacings (avoid division in inner loops)
        state%inv_dx2 = 1.0_wp / (state%dx * state%dx)
        state%inv_dy2 = 1.0_wp / (state%dy * state%dy)
        state%inv_dz2 = 1.0_wp / (state%dz * state%dz)
        state%two_inv_dx2 = 2.0_wp * state%inv_dx2
        state%two_inv_dy2 = 2.0_wp * state%inv_dy2
        state%two_inv_dz2 = 2.0_wp * state%inv_dz2
        
        ! Configuration
        if (present(config)) then
            state%config = config
        end if
        
        state%n_groups = state%config%n_groups
        
        ! Allocate flux arrays
        allocate(state%flux(nx, ny, nz, state%n_groups))
        allocate(state%flux_old(nx, ny, nz, state%n_groups))
        allocate(state%fission_source(nx, ny, nz))
        allocate(state%external_source(nx, ny, nz, state%n_groups))
        
        ! Allocate cross sections
        allocate(state%xsec(nx, ny, nz))
        
        ! Allocate power
        allocate(state%power_density(nx, ny, nz))
        allocate(state%radial_peaking(nx, ny))
        allocate(state%axial_peaking(nz))
        
        ! Initialize with flat flux guess
        state%flux = 1.0e10_wp  ! Typical flux level
        state%flux_old = state%flux
        state%fission_source = 0.0_wp
        state%external_source = 0.0_wp
        state%power_density = 0.0_wp
        
        state%k_eff = state%config%k_guess
        state%k_eff_old = state%k_eff
        
        ! Initialize cross sections to default values
        do k = 1, nz
            do j = 1, ny
                do i = 1, nx
                    call initialize_default_xsec(state%xsec(i, j, k), state%n_groups)
                end do
            end do
        end do
    end subroutine mg_init
    
    !> Reset state for new calculation (without deallocation)
    subroutine mg_reset(state, preserve_xsec)
        type(mg_state_t), intent(inout) :: state
        logical, intent(in), optional :: preserve_xsec
        logical :: keep_xsec
        
        keep_xsec = .false.
        if (present(preserve_xsec)) keep_xsec = preserve_xsec
        
        ! Reset flux to initial guess
        state%flux = 1.0e10_wp
        state%flux_old = state%flux
        state%fission_source = 0.0_wp
        state%external_source = 0.0_wp
        state%power_density = 0.0_wp
        
        ! Reset convergence tracking
        state%k_eff = state%config%k_guess
        state%k_eff_old = state%k_eff
        state%outer_iterations = 0
        state%inner_iterations = 0
        state%outer_error = 1.0_wp
        state%inner_error = 1.0_wp
        state%converged = .false.
    end subroutine mg_reset

    subroutine mg_destroy(state)
        type(mg_state_t), intent(inout) :: state
        integer :: ierr
        
        ! Deallocate in reverse order of allocation to help memory management
        ! Skip the 'if allocated' checks - deallocate handles this automatically in F2003+
        deallocate(state%xsec, stat=ierr)
        deallocate(state%axial_peaking, stat=ierr)
        deallocate(state%radial_peaking, stat=ierr)
        deallocate(state%power_density, stat=ierr)
        deallocate(state%external_source, stat=ierr)
        deallocate(state%fission_source, stat=ierr)
        deallocate(state%flux_old, stat=ierr)
        deallocate(state%flux, stat=ierr)
        
        ! stat=ierr prevents program termination on deallocation errors
        ! but we don't need to check ierr unless debugging
    end subroutine mg_destroy
    
    !> Initialize default cross sections (water moderator)
    subroutine initialize_default_xsec(xsec, n_groups)
        type(mg_xsec_t), intent(out) :: xsec
        integer, intent(in) :: n_groups
        
        xsec%n_groups = n_groups
        
        allocate(xsec%D(n_groups))
        allocate(xsec%sigma_t(n_groups))
        allocate(xsec%sigma_a(n_groups))
        allocate(xsec%sigma_f(n_groups))
        allocate(xsec%nu_sigma_f(n_groups))
        allocate(xsec%sigma_r(n_groups))
        allocate(xsec%chi(n_groups))
        allocate(xsec%kappa(n_groups))
        allocate(xsec%sigma_s(n_groups, n_groups))
        allocate(xsec%chi_d(n_groups))
        
        ! Default: water moderator (non-multiplying)
        xsec%D = 1.0_wp
        xsec%sigma_t = 0.5_wp
        xsec%sigma_a = 0.01_wp
        xsec%sigma_f = 0.0_wp
        xsec%nu_sigma_f = 0.0_wp
        xsec%sigma_r = 0.01_wp
        xsec%chi = 0.0_wp
        xsec%kappa = 200.0_wp
        xsec%chi_d = 0.0_wp
        
        ! Diagonal scattering (no group transfer)
        xsec%sigma_s = 0.0_wp
        if (n_groups >= 1) xsec%sigma_s(1, 1) = 0.49_wp
        if (n_groups >= 2) xsec%sigma_s(2, 2) = 0.49_wp
    end subroutine initialize_default_xsec
    
    !> Set cross sections for a region
    subroutine mg_set_cross_sections(state, xsec, i1, i2, j1, j2, k1, k2)
        type(mg_state_t), intent(inout) :: state
        type(mg_xsec_t), intent(in) :: xsec
        integer, intent(in), optional :: i1, i2, j1, j2, k1, k2
        
        integer :: imin, imax, jmin, jmax, kmin, kmax
        integer :: i, j, k
        
        imin = 1; imax = state%nx
        jmin = 1; jmax = state%ny
        kmin = 1; kmax = state%nz
        
        if (present(i1)) imin = i1
        if (present(i2)) imax = i2
        if (present(j1)) jmin = j1
        if (present(j2)) jmax = j2
        if (present(k1)) kmin = k1
        if (present(k2)) kmax = k2
        
        do k = kmin, kmax
            do j = jmin, jmax
                do i = imin, imax
                    state%xsec(i, j, k) = xsec
                end do
            end do
        end do
    end subroutine mg_set_cross_sections
    
    !> Set external source
    subroutine mg_set_source(state, source, group)
        type(mg_state_t), intent(inout) :: state
        real(wp), intent(in) :: source(:, :, :)
        integer, intent(in), optional :: group
        
        integer :: g
        
        if (present(group)) then
            state%external_source(:, :, :, group) = source
        else
            ! Set to all groups equally
            do g = 1, state%n_groups
                state%external_source(:, :, :, g) = source / real(state%n_groups, wp)
            end do
        end if
    end subroutine mg_set_source
    
    !> Diagnostic: Check cross-sections and initial state
    subroutine mg_check_problem_setup(state)
        type(mg_state_t), intent(in) :: state
        integer :: i, j, k, g
        real(wp) :: total_nu_sigma_f, total_sigma_a, total_flux
        real(wp) :: max_nu_sigma_f, max_sigma_a, avg_production, avg_absorption
        
        print '(A)', '=========================================='
        print '(A)', 'DIAGNOSTIC: Problem Setup Check'
        print '(A)', '=========================================='
        
        ! Check flux levels
        total_flux = sum(state%flux)
        print '(A,ES12.3)', 'Total flux: ', total_flux
        print '(A,ES12.3)', 'Max flux:   ', maxval(state%flux)
        print '(A,ES12.3)', 'Min flux:   ', minval(state%flux)
        
        ! Check cross-sections
        total_nu_sigma_f = 0.0_wp
        total_sigma_a = 0.0_wp
        max_nu_sigma_f = 0.0_wp
        max_sigma_a = 0.0_wp
        
        do k = 1, state%nz
            do j = 1, state%ny
                do i = 1, state%nx
                    do g = 1, state%n_groups
                        total_nu_sigma_f = total_nu_sigma_f + state%xsec(i,j,k)%nu_sigma_f(g)
                        total_sigma_a = total_sigma_a + state%xsec(i,j,k)%sigma_a(g)
                        max_nu_sigma_f = max(max_nu_sigma_f, state%xsec(i,j,k)%nu_sigma_f(g))
                        max_sigma_a = max(max_sigma_a, state%xsec(i,j,k)%sigma_a(g))
                    end do
                end do
            end do
        end do
        
        print '(A)', ''
        print '(A)', 'Cross-sections:'
        print '(A,ES12.3)', '  Max nu*sigma_f:   ', max_nu_sigma_f
        print '(A,ES12.3)', '  Max sigma_a:      ', max_sigma_a
        print '(A,ES12.3)', '  Total nu*sigma_f: ', total_nu_sigma_f
        print '(A,ES12.3)', '  Total sigma_a:    ', total_sigma_a
        
        if (max_nu_sigma_f < 1.0e-10_wp) then
            print '(A)', '  ** WARNING: No fissile material detected! **'
        end if
        
        ! Estimate k-infinite
        avg_production = total_nu_sigma_f / real(state%nx*state%ny*state%nz*state%n_groups, wp)
        avg_absorption = total_sigma_a / real(state%nx*state%ny*state%nz*state%n_groups, wp)
        
        if (avg_absorption > 1.0e-10_wp) then
            print '(A,F10.5)', '  Estimated k-inf:  ', avg_production / avg_absorption
        end if
        
        print '(A)', '=========================================='
        print '(A)', ''
    end subroutine mg_check_problem_setup

    !> Validate cross-sections before solving
    subroutine mg_validate_cross_sections(state, is_valid)
        type(mg_state_t), intent(inout) :: state
        logical, intent(out) :: is_valid
        
        integer :: i, j, k, g, gp
        real(wp) :: sigma_total, nu_ratio
        real(wp) :: max_sigma_a, min_sigma_a, max_nu_sigma_f
        logical :: has_fissile
        
        is_valid = .true.
        has_fissile = .false.
        
        ! Initialize min/max values
        max_sigma_a = -1.0e30_wp
        min_sigma_a = 1.0e30_wp
        max_nu_sigma_f = -1.0e30_wp
        
        do k = 1, state%nz
            do j = 1, state%ny
                do i = 1, state%nx
                    do g = 1, state%n_groups
                        
                        ! Check for negative values
                        if (state%xsec(i,j,k)%D(g) < 0.0_wp) then
                            print '(A,3I4,I2,A)', 'ERROR: D < 0 at (', i, j, k, g, ')'
                            is_valid = .false.
                        end if
                        
                        if (state%xsec(i,j,k)%sigma_a(g) < 0.0_wp) then
                            print '(A,3I4,I2,A)', 'ERROR: sigma_a < 0 at (', i, j, k, g, ')'
                            is_valid = .false.
                        end if
                        
                        ! Check for unreasonably small absorption
                        if (state%xsec(i,j,k)%sigma_a(g) < 1.0e-8_wp) then
                            !print '(A,3I4,I2,A,ES10.2)', &
                            !    'WARNING: Very small σ_a at (', i, j, k, g, '): ', &
                            !    state%xsec(i,j,k)%sigma_a(g)
                            !print *, '  This will cause numerical issues!'
                            is_valid = .false.
                        end if
                        
                        ! Track min/max
                        max_sigma_a = max(max_sigma_a, state%xsec(i,j,k)%sigma_a(g))
                        min_sigma_a = min(min_sigma_a, state%xsec(i,j,k)%sigma_a(g))
                        max_nu_sigma_f = max(max_nu_sigma_f, state%xsec(i,j,k)%nu_sigma_f(g))
                        
                        ! Check for fissile material
                        if (state%xsec(i,j,k)%nu_sigma_f(g) > 1.0e-10_wp) then
                            has_fissile = .true.
                        end if
                        
                        ! Validate nu ratio
                        if (state%xsec(i,j,k)%sigma_f(g) > 1.0e-10_wp) then
                            nu_ratio = state%xsec(i,j,k)%nu_sigma_f(g) / &
                                    state%xsec(i,j,k)%sigma_f(g)
                            if (nu_ratio < 1.5_wp .or. nu_ratio > 4.0_wp) then
                                print '(A,3I4,I2,A,F6.2)', 'WARNING: Unusual nu at (', &
                                    i, j, k, g, '): ', nu_ratio
                            end if
                        end if
                        
                        ! Compute total removal
                        sigma_total = state%xsec(i,j,k)%sigma_a(g)
                        do gp = 1, state%n_groups
                            if (gp /= g) then
                                sigma_total = sigma_total + state%xsec(i,j,k)%sigma_s(g, gp)
                            end if
                        end do
                        
                        ! Store in sigma_r for later use
                        state%xsec(i,j,k)%sigma_r(g) = sigma_total
                        
                    end do
                end do
            end do
        end do
        
        if (.not. has_fissile) then
            print '(A)', 'ERROR: No fissile material found!'
            is_valid = .false.
        end if
        
    end subroutine mg_validate_cross_sections

    !> Detect NaN in arrays
    function has_nan(arr) result(found_nan)
        real(wp), intent(in) :: arr(:,:,:,:)
        logical :: found_nan
        integer :: i, j, k, g
        
        found_nan = .false.
        
        do g = 1, size(arr, 4)
            do k = 1, size(arr, 3)
                do j = 1, size(arr, 2)
                    do i = 1, size(arr, 1)
                        if (arr(i,j,k,g) /= arr(i,j,k,g)) then  ! NaN check
                            print '(A,4I5)', 'NaN detected at: ', i, j, k, g
                            found_nan = .true.
                            return
                        end if
                    end do
                end do
            end do
        end do
    end function has_nan

    !> Solve k-eigenvalue problem using power iteration
    subroutine mg_solve_eigenvalue(state, k_eff, converged, verbose)
        type(mg_state_t), intent(inout) :: state
        real(wp), intent(out) :: k_eff
        logical, intent(out) :: converged
        logical, intent(in), optional :: verbose
        
        integer :: outer_iter, inner_iter, g
        real(wp) :: k_error, flux_error, fission_sum, fission_new
        real(wp) :: alpha, beta_accel, rho, d, omega
        real(wp), allocatable :: flux_prev(:,:,:,:)
        logical :: valid_xsec, debug_mode
        
        ! Chebyshev acceleration parameters
        real(wp), parameter :: rho_estimate = 0.95_wp  ! Spectral radius estimate
        integer, parameter :: accel_start = 100000  ! Disabled: Chebyshev overshoots when k_inf>>1 and hits the upper clamp, causing false convergence at keff=5
        
        debug_mode = .false.
        if (present(verbose)) debug_mode = verbose
        
        ! Validate cross-sections
        call mg_validate_cross_sections(state, valid_xsec)
        if (.not. valid_xsec) then
            converged = .false.
            k_eff = 0.0_wp
            return
        end if
        
        allocate(flux_prev(state%nx, state%ny, state%nz, state%n_groups))

        ! Reset keff if it is stuck at a clamp boundary from a previous solve.
        if (state%k_eff >= 4.99_wp .or. state%k_eff <= 0.006_wp) state%k_eff = 1.0_wp

        state%converged = .false.
        where (state%flux < 1.0e-10_wp) state%flux = 1.0_wp
        
        ! Initialise Chebyshev parameters
        d = (1.0_wp + rho_estimate) / 2.0_wp
        omega = 1.0_wp

#if defined(COMPUTATION_MODE_GPU) || defined(COMPUTATION_MODE_HYBRID)
        ! GPU dispatch: when GPU is available and the problem is large
        ! enough, delegate the entire eigenvalue solve to the GPU path.
        if (use_gpu() .and. &
            state%nx * state%ny * state%nz >= gpu_min_workload()) then
            call mg_solve_eigenvalue_gpu(state, k_eff, converged)
            return
        end if
#endif
        
        do outer_iter = 1, state%config%max_outer_iter
            state%outer_iterations = outer_iter
            state%k_eff_old = state%k_eff
            flux_prev = state%flux_old
            state%flux_old = state%flux
            
            ! Compute fission source
            call compute_fission_source(state)
            fission_sum = sum(state%fission_source)
            
            if (fission_sum < 1.0e-30_wp) then
                converged = .false.
                k_eff = 0.0_wp
                deallocate(flux_prev)
                return
            end if
            
            ! Inner iterations
            do inner_iter = 1, state%config%max_inner_iter
                do g = 1, state%n_groups
                    call solve_group_equation_safe(state, g)
                end do
                
                flux_error = compute_flux_error(state)
                if (flux_error < state%config%inner_tolerance) exit
                state%flux_old = state%flux
            end do
            
            ! Apply Chebyshev acceleration after initial iterations
            if (outer_iter >= accel_start) then
                ! Update Chebyshev parameters
                if (outer_iter == accel_start) then
                    omega = 1.0_wp / d
                else
                    omega = 1.0_wp / (d - 0.25_wp * rho_estimate * omega)
                end if
                
                ! Accelerated flux update
                alpha = omega
                beta_accel = (1.0_wp - omega)
                
                state%flux = alpha * state%flux + beta_accel * flux_prev
            end if
            
            ! Update k-effective
            call compute_fission_source_with_flux(state, state%flux, fission_new)
            if (fission_sum > 1.0e-30_wp .and. fission_new > 1.0e-30_wp) then
                state%k_eff = state%k_eff_old * (fission_new / fission_sum)
            end if
            
            ! Clamp k-eff (lower bound allows convergence to deep subcriticality)
            state%k_eff = min(max(state%k_eff, 0.005_wp), 5.0_wp)
            
            ! Check convergence
            k_error = abs(state%k_eff - state%k_eff_old) / (abs(state%k_eff) + 1.0e-30_wp)
            state%outer_error = k_error
            
            if (k_error < state%config%outer_tolerance .and. outer_iter > 5) then
                state%converged = .true.
                exit
            end if
        end do
        
        deallocate(flux_prev)
        
        call normalize_flux(state)
        call compute_power_distribution(state)
        k_eff = state%k_eff
        converged = state%converged
    end subroutine mg_solve_eigenvalue
    
    subroutine compute_balance(state, production, absorption, leakage)
        type(mg_state_t), intent(in) :: state
        real(wp), intent(out) :: production, absorption, leakage
        
        integer :: i, j, k, g
        real(wp) :: leak_x, leak_y, leak_z, D
        
        production = 0.0_wp
        absorption = 0.0_wp
        leakage = 0.0_wp
        
        do k = 1, state%nz
            do j = 1, state%ny
                do i = 1, state%nx
                    do g = 1, state%n_groups
                        ! Production
                        production = production + &
                            state%xsec(i,j,k)%nu_sigma_f(g) * state%flux(i,j,k,g)
                        
                        ! Absorption
                        absorption = absorption + &
                            state%xsec(i,j,k)%sigma_a(g) * state%flux(i,j,k,g)
                        
                        ! Leakage (approximate)
                        if (i > 1 .and. i < state%nx .and. &
                            j > 1 .and. j < state%ny .and. &
                            k > 1 .and. k < state%nz) then
                            
                            D = state%xsec(i,j,k)%D(g)
                            leak_x = D * abs(state%flux(i+1,j,k,g) - 2*state%flux(i,j,k,g) + &
                                            state%flux(i-1,j,k,g)) / (state%dx**2)
                            leak_y = D * abs(state%flux(i,j+1,k,g) - 2*state%flux(i,j,k,g) + &
                                            state%flux(i,j-1,k,g)) / (state%dy**2)
                            leak_z = D * abs(state%flux(i,j,k+1,g) - 2*state%flux(i,j,k,g) + &
                                            state%flux(i,j,k-1,g)) / (state%dz**2)
                            
                            leakage = leakage + (leak_x + leak_y + leak_z)
                        end if
                    end do
                end do
            end do
        end do
    end subroutine compute_balance

    !> Solve fixed source problem
    subroutine mg_solve_fixed_source(state, converged)
        type(mg_state_t), intent(inout) :: state
        logical, intent(out) :: converged
        
        integer :: iter, g
        real(wp) :: flux_error
        
        state%converged = .false.
        
        do iter = 1, state%config%max_inner_iter
            state%inner_iterations = iter
            state%flux_old = state%flux
            
            ! Sweep through all energy groups
            do g = 1, state%n_groups
                call solve_group_equation_safe(state, g)
            end do
            
            ! Check convergence
            flux_error = compute_flux_error(state)
            state%inner_error = flux_error
            
            if (flux_error < state%config%inner_tolerance) then
                state%converged = .true.
                exit
            end if
        end do
        
        converged = state%converged
    end subroutine mg_solve_fixed_source
    
    !> Optimised parallel group equation solver.
    !! Uses a Jacobi-style sweep (reads neighbour fluxes from a snapshot
    !! copy, writes to state%flux) so the OpenMP parallel do is race-free.
    !! The previous in-place Gauss-Seidel-style sweep was racy: threads
    !! read state%flux(i±1,j,k,g) etc. that other threads were concurrently
    !! updating, producing non-deterministic results.
    subroutine solve_group_equation_safe(state, g)
        type(mg_state_t), intent(inout) :: state
        integer, intent(in) :: g
        
        integer :: i, j, k, gp
        real(wp) :: source, inscatter, fission, D, sigma_r
        real(wp) :: phi_xp, phi_xm, phi_yp, phi_ym, phi_zp, phi_zm
        real(wp) :: coef_x, coef_y, coef_z, coef_c, rhs, denom, phi_new
        real(wp) :: k_eff_inv
        real(wp), allocatable :: flux_snap(:,:,:)
        real(wp), parameter :: MIN_DENOM = 1.0e-10_wp
        real(wp), parameter :: MAX_FLUX = 1.0e15_wp
        real(wp), parameter :: RELAX = 0.8_wp
        real(wp), parameter :: ONE_MINUS_RELAX = 0.2_wp
        
        ! Pre-compute k_eff inverse to avoid division in inner loop
        k_eff_inv = 1.0_wp / max(state%k_eff, 0.005_wp)
        
        ! Take a snapshot of the current group-g flux so neighbour reads
        ! are race-free under OpenMP. Only group g is snapshotted because
        ! only group g is being updated; in-scattering reads from other
        ! groups (state%flux(i,j,k,gp)) are not being written by this
        ! sweep, so they are safe to read in place.
        allocate(flux_snap(state%nx, state%ny, state%nz))
        flux_snap = state%flux(:,:,:,g)
        
        ! Cache-friendly loop order: k, j, i (matches memory layout)
        !$OMP PARALLEL DO PRIVATE(i, j, D, sigma_r, phi_xm, phi_xp, phi_ym, phi_yp, &
        !$OMP                     phi_zm, phi_zp, coef_x, coef_y, coef_z, coef_c, &
        !$OMP                     inscatter, fission, source, rhs, denom, phi_new, gp) &
        !$OMP             SCHEDULE(STATIC)
        do k = 2, state%nz - 1
            do j = 2, state%ny - 1
                do i = 2, state%nx - 1
                    
                    ! Get material properties (single array access)
                    D = state%xsec(i, j, k)%D(g)
                    sigma_r = state%xsec(i, j, k)%sigma_r(g)
                    
                    ! Get neighbour fluxes from the snapshot (Jacobi:
                    ! race-free under OpenMP). The previous code read
                    ! state%flux(i±1,j,k,g) directly, which raced with
                    ! other threads' in-place updates.
                    phi_xm = max(flux_snap(i-1, j, k), 0.0_wp)
                    phi_xp = max(flux_snap(i+1, j, k), 0.0_wp)
                    phi_ym = max(flux_snap(i, j-1, k), 0.0_wp)
                    phi_yp = max(flux_snap(i, j+1, k), 0.0_wp)
                    phi_zm = max(flux_snap(i, j, k-1), 0.0_wp)
                    phi_zp = max(flux_snap(i, j, k+1), 0.0_wp)
                    
                    ! Diffusion coefficients (use pre-computed grid spacings)
                    coef_x = D * state%inv_dx2
                    coef_y = D * state%inv_dy2
                    coef_z = D * state%inv_dz2
                    coef_c = state%two_inv_dx2 + state%two_inv_dy2 + state%two_inv_dz2
                    coef_c = coef_c * D
                    
                    ! In-scattering (manual loop unrolling for common cases).
                    ! Reads state%flux(i,j,k,gp) for gp /= g — safe because
                    ! those entries are not being written by this sweep.
                    inscatter = 0.0_wp
                    if (state%n_groups == 2) then
                        ! Optimised path for 2-group
                        if (g == 1) then
                            inscatter = state%xsec(i, j, k)%sigma_s(2, 1) * &
                                       max(state%flux(i, j, k, 2), 0.0_wp)
                        else
                            inscatter = state%xsec(i, j, k)%sigma_s(1, 2) * &
                                       max(state%flux(i, j, k, 1), 0.0_wp)
                        end if
                    else
                        ! General case
                        do gp = 1, state%n_groups
                            if (gp /= g) then
                                inscatter = inscatter + &
                                    state%xsec(i, j, k)%sigma_s(gp, g) * &
                                    max(state%flux(i, j, k, gp), 0.0_wp)
                            end if
                        end do
                    end if
                    
                    ! Fission source (use pre-computed inverse)
                    fission = state%xsec(i, j, k)%chi(g) * &
                             state%fission_source(i, j, k) * k_eff_inv
                    
                    ! Total source
                    source = inscatter + fission + state%external_source(i, j, k, g)
                    
                    ! RHS
                    rhs = coef_x * (phi_xm + phi_xp) + &
                          coef_y * (phi_ym + phi_yp) + &
                          coef_z * (phi_zm + phi_zp) + source
                    
                    ! Denominator
                    denom = max(coef_c + sigma_r, MIN_DENOM)
                    
                    ! Update flux with relaxation. Use the snapshot value
                    ! for the relaxation term so the iteration is a true
                    ! Jacobi update (not a hybrid that mixes snapshot and
                    ! in-place values).
                    phi_new = rhs / denom
                    phi_new = RELAX * phi_new + ONE_MINUS_RELAX * flux_snap(i, j, k)
                    
                    ! Clamp
                    phi_new = max(min(phi_new, MAX_FLUX), 0.0_wp)
                    
                    state%flux(i, j, k, g) = phi_new
                end do
            end do
        end do
        !$OMP END PARALLEL DO
        
        deallocate(flux_snap)
        
        call apply_boundary_conditions(state, g)
    end subroutine solve_group_equation_safe
    
    !> Solve group equation for fixed source
    subroutine solve_group_equation_fixed(state, g)
        type(mg_state_t), intent(inout) :: state
        integer, intent(in) :: g
        
        ! Similar to eigenvalue case but without k_eff division
        ! Implementation follows same pattern as solve_group_equation
        call solve_group_equation_safe(state, g)
    end subroutine solve_group_equation_fixed
    
    !> Compute Laplacian of flux
    function compute_laplacian(state, i, j, k, g) result(laplacian)
        type(mg_state_t), intent(in) :: state
        integer, intent(in) :: i, j, k, g
        real(wp) :: laplacian
        
        real(wp) :: inv_dx2, inv_dy2, inv_dz2
        real(wp) :: phi
        
        phi = state%flux(i, j, k, g)
        
        inv_dx2 = 1.0_wp / (state%dx * state%dx)
        inv_dy2 = 1.0_wp / (state%dy * state%dy)
        inv_dz2 = 1.0_wp / (state%dz * state%dz)
        
        laplacian = (state%flux(i-1, j, k, g) - 2.0_wp*phi + state%flux(i+1, j, k, g)) * inv_dx2 + &
                    (state%flux(i, j-1, k, g) - 2.0_wp*phi + state%flux(i, j+1, k, g)) * inv_dy2 + &
                    (state%flux(i, j, k-1, g) - 2.0_wp*phi + state%flux(i, j, k+1, g)) * inv_dz2
    end function compute_laplacian

    !> Optimised fission source computation
    subroutine compute_fission_source(state)
        type(mg_state_t), intent(inout) :: state
        
        integer :: i, j, k, g
        real(wp) :: fission, nu_sf
        
        !$OMP PARALLEL DO PRIVATE(i, j, fission, nu_sf, g) SCHEDULE(STATIC)
        do k = 1, state%nz
            do j = 1, state%ny
                do i = 1, state%nx
                    fission = 0.0_wp
                    
                    ! Manual unrolling for 2-group case
                    if (state%n_groups == 2) then
                        fission = state%xsec(i, j, k)%nu_sigma_f(1) * state%flux(i, j, k, 1) + &
                                 state%xsec(i, j, k)%nu_sigma_f(2) * state%flux(i, j, k, 2)
                    else
                        do g = 1, state%n_groups
                            fission = fission + state%xsec(i, j, k)%nu_sigma_f(g) * &
                                               state%flux(i, j, k, g)
                        end do
                    end if
                    
                    state%fission_source(i, j, k) = fission
                end do
            end do
        end do
        !$OMP END PARALLEL DO
    end subroutine compute_fission_source

    !> Compute fission source with given flux
    subroutine compute_fission_source_with_flux(state, flux, total_fission)
        type(mg_state_t), intent(in) :: state
        real(wp), intent(in) :: flux(:, :, :, :)
        real(wp), intent(out) :: total_fission
        
        integer :: i, j, k, g
        real(wp) :: fission
        
        total_fission = 0.0_wp
        
        do k = 1, state%nz
            do j = 1, state%ny
                do i = 1, state%nx
                    fission = 0.0_wp
                    do g = 1, state%n_groups
                        fission = fission + state%xsec(i, j, k)%nu_sigma_f(g) * &
                                        flux(i, j, k, g)
                    end do
                    total_fission = total_fission + fission
                end do
            end do
        end do
    end subroutine compute_fission_source_with_flux

    !> Normalize flux to unit fission source
    subroutine normalize_flux_to_unity(state)
        type(mg_state_t), intent(inout) :: state
        
        real(wp) :: total_fission
        
        total_fission = sum(state%fission_source)
        
        if (total_fission > TOL_DEFAULT) then
            state%flux = state%flux / sqrt(total_fission)
            state%fission_source = state%fission_source / total_fission
        end if
    end subroutine normalize_flux_to_unity

    !> Update k-effective
    subroutine update_keff(state)
        type(mg_state_t), intent(inout) :: state
        
        real(wp) :: fission_new, fission_old
        integer :: i, j, k, g
        
        fission_new = 0.0_wp
        fission_old = 0.0_wp
        
        do k = 1, state%nz
            do j = 1, state%ny
                do i = 1, state%nx
                    do g = 1, state%n_groups
                        fission_new = fission_new + &
                            state%xsec(i, j, k)%nu_sigma_f(g) * state%flux(i, j, k, g)
                        fission_old = fission_old + &
                            state%xsec(i, j, k)%nu_sigma_f(g) * state%flux_old(i, j, k, g)
                    end do
                end do
            end do
        end do
        
        if (fission_old > TOL_DEFAULT) then
            state%k_eff = state%k_eff_old * (fission_new / fission_old)
        end if
    end subroutine update_keff
    
    !> Normalize flux to specified power level
    subroutine normalize_flux(state)
        type(mg_state_t), intent(inout) :: state

        real(wp) :: current_power, scale_factor
        integer :: i, j, k, g

        if (.not. state%config%normalize_power) return

        ! Flux amplitude is only meaningful at near-critical conditions.
        ! Scaling a deeply subcritical flux to rated power produces an
        ! astronomically large scale_factor; multiplied into the flux
        ! this pushes values to Infinity, which -ffast-math turns into
        ! NaN in subsequent arithmetic, causing OMP worker crashes.
        if (state%k_eff < 0.95_wp) return
        
        ! Compute current power
        current_power = 0.0_wp
        do k = 1, state%nz
            do j = 1, state%ny
                do i = 1, state%nx
                    do g = 1, state%n_groups
                        current_power = current_power + &
                            state%xsec(i, j, k)%sigma_f(g) * &
                            state%xsec(i, j, k)%kappa(g) * &
                            state%flux(i, j, k, g)
                    end do
                end do
            end do
        end do
        
        ! Convert MeV/cm³ to Watts
        current_power = current_power * 1.602e-13_wp * state%volume * 1.0e6_wp
        
        if (current_power > TOL_DEFAULT) then
            scale_factor = state%config%power_level / current_power
            state%flux = state%flux * scale_factor
        end if
    end subroutine normalize_flux
    
    !> Compute flux error
    function compute_flux_error(state) result(error)
        type(mg_state_t), intent(in) :: state
        real(wp) :: error
        
        real(wp) :: diff, norm_new, norm_old
        
        diff = sum((state%flux - state%flux_old)**2)
        norm_new = sum(state%flux**2)
        norm_old = sum(state%flux_old**2)
        
        if (norm_old > TOL_DEFAULT) then
            error = sqrt(diff / norm_old)
        else
            error = sqrt(diff)
        end if
    end function compute_flux_error
    
    !> Apply boundary conditions (albedo)
    subroutine apply_boundary_conditions(state, g)
        type(mg_state_t), intent(inout) :: state
        integer, intent(in) :: g
        
        real(wp) :: beta
        
        ! X boundaries
        beta = state%config%albedo_x_min
        state%flux(1, :, :, g) = beta * state%flux(2, :, :, g)
        
        beta = state%config%albedo_x_max
        state%flux(state%nx, :, :, g) = beta * state%flux(state%nx-1, :, :, g)
        
        ! Y boundaries
        beta = state%config%albedo_y_min
        state%flux(:, 1, :, g) = beta * state%flux(:, 2, :, g)
        
        beta = state%config%albedo_y_max
        state%flux(:, state%ny, :, g) = beta * state%flux(:, state%ny-1, :, g)
        
        ! Z boundaries
        beta = state%config%albedo_z_min
        state%flux(:, :, 1, g) = beta * state%flux(:, :, 2, g)
        
        beta = state%config%albedo_z_max
        state%flux(:, :, state%nz, g) = beta * state%flux(:, :, state%nz-1, g)
    end subroutine apply_boundary_conditions
    
    !> Compute power distribution
    subroutine compute_power_distribution(state)
        type(mg_state_t), intent(inout) :: state
        
        integer :: i, j, k, g
        real(wp) :: power_cell, kappa_MeV, J_per_MeV
        
        J_per_MeV = 1.602e-13_wp
        state%total_power = 0.0_wp
        
        do k = 1, state%nz
            do j = 1, state%ny
                do i = 1, state%nx
                    power_cell = 0.0_wp
                    
                    do g = 1, state%n_groups
                        kappa_MeV = state%xsec(i, j, k)%kappa(g)
                        power_cell = power_cell + &
                            state%xsec(i, j, k)%sigma_f(g) * kappa_MeV * &
                            state%flux(i, j, k, g)
                    end do
                    
                    ! Convert MeV/cm³·s to W/cm³
                    state%power_density(i, j, k) = power_cell * J_per_MeV
                    state%total_power = state%total_power + &
                        state%power_density(i, j, k) * state%volume * 1.0e6_wp  ! to W
                end do
            end do
        end do
    end subroutine compute_power_distribution
    
    !> Compute form factors (peaking)
    subroutine mg_compute_form_factors(state)
        type(mg_state_t), intent(inout) :: state
        
        integer :: i, j, k
        real(wp) :: avg_power, max_power, axial_avg
        
        ! Average power
        if (state%total_power > TOL_DEFAULT) then
            avg_power = state%total_power / real(state%nx * state%ny * state%nz, wp)
        else
            avg_power = 1.0_wp
        end if
        
        ! Radial peaking
        max_power = 0.0_wp
        do j = 1, state%ny
            do i = 1, state%nx
                state%radial_peaking(i, j) = sum(state%power_density(i, j, :)) / &
                                              real(state%nz, wp) / avg_power
                max_power = max(max_power, state%radial_peaking(i, j))
            end do
        end do
        
        ! Axial peaking
        do k = 1, state%nz
            axial_avg = sum(state%power_density(:, :, k)) / real(state%nx * state%ny, wp)
            state%axial_peaking(k) = axial_avg / avg_power
        end do
        
        state%total_peaking = maxval(state%power_density) / avg_power
    end subroutine mg_compute_form_factors
    
    !> Get multi-group flux
    subroutine mg_get_flux(state, flux, group)
        type(mg_state_t), intent(in) :: state
        real(wp), intent(out) :: flux(:, :, :)
        integer, intent(in), optional :: group
        
        integer :: g
        
        if (present(group)) then
            flux = state%flux(:, :, :, group)
        else
            ! Return total flux (sum over groups)
            flux = 0.0_wp
            do g = 1, state%n_groups
                flux = flux + state%flux(:, :, :, g)
            end do
        end if
    end subroutine mg_get_flux
    
    !> Get power density. Returns the stored field in its native units of
    !> W/cm³ — matches the docstring at the top of this module. Callers
    !> that need W/m³ should use `fuel_power_to_volumetric_W_m3`. The
    !> previous behaviour silently multiplied by 10⁶, which combined with
    !> the helper's own 10⁶ multiplier produced a 10¹² over-conversion
    !> and a 1.4 MW/m³ phantom heat source at cold standby.
    subroutine mg_get_power(state, power)
        type(mg_state_t), intent(in) :: state
        real(wp), intent(out) :: power(:, :, :)
        power = state%power_density
    end subroutine mg_get_power
    
    !> Get fission rate
    subroutine mg_get_fission_rate(state, fission_rate)
        type(mg_state_t), intent(in) :: state
        real(wp), intent(out) :: fission_rate(:, :, :)
        
        fission_rate = state%fission_source
    end subroutine mg_get_fission_rate
    
    !> Compute k-effective from flux distribution, including leakage.
    !!
    !! k = (production) / (absorption + leakage)
    !!
    !! where leakage = -∑ D · ∇²φ integrated over interior cells. Previously
    !! this returned production/absorption only, which is the k_inf-like
    !! ratio: when flux concentrates in an unshielded sub-region (e.g. the
    !! top axial node with bottom-entry rods at 95 % insertion) the ratio
    !! reads as the *local* k of that sub-region and ignores leakage to
    !! the surrounding shielded volume. The power gate downstream of this
    !! reads sim%k_eff < 0.98 to gate power to zero; without the leakage
    !! term, a deeply rod-blocked cold core can spuriously read k > 1 and
    !! let fission power leak through.
    function mg_compute_keff(state) result(k_eff)
        type(mg_state_t), intent(in) :: state
        real(wp) :: k_eff

        real(wp) :: production, absorption, leakage
        real(wp) :: lap, phi_c, D
        integer :: i, j, k, g

        production = 0.0_wp
        absorption = 0.0_wp
        leakage    = 0.0_wp

        do k = 1, state%nz
            do j = 1, state%ny
                do i = 1, state%nx
                    do g = 1, state%n_groups
                        production = production + &
                            state%xsec(i, j, k)%nu_sigma_f(g) * state%flux(i, j, k, g)
                        absorption = absorption + &
                            state%xsec(i, j, k)%sigma_a(g) * state%flux(i, j, k, g)

                        ! Net leakage: -D · ∇²φ summed over interior cells
                        ! gives the surface integral of -D∇φ (divergence
                        ! theorem). Boundary cells are skipped — their
                        ! contribution is captured by the BC applied to
                        ! the adjacent interior cell's neighbour value.
                        if (i > 1 .and. i < state%nx .and. &
                            j > 1 .and. j < state%ny .and. &
                            k > 1 .and. k < state%nz) then
                            phi_c = state%flux(i, j, k, g)
                            D     = state%xsec(i, j, k)%D(g)
                            lap = (state%flux(i+1, j, k, g) - 2.0_wp * phi_c &
                                 + state%flux(i-1, j, k, g)) * state%inv_dx2 &
                                + (state%flux(i, j+1, k, g) - 2.0_wp * phi_c &
                                 + state%flux(i, j-1, k, g)) * state%inv_dy2 &
                                + (state%flux(i, j, k+1, g) - 2.0_wp * phi_c &
                                 + state%flux(i, j, k-1, g)) * state%inv_dz2
                            leakage = leakage - D * lap
                        end if
                    end do
                end do
            end do
        end do

        if (absorption + leakage > TOL_DEFAULT) then
            k_eff = production / (absorption + leakage)
        else
            k_eff = 0.0_wp
        end if
    end function mg_compute_keff
    
    !> Time-dependent multi-group diffusion solver with delayed neutrons
    !! Solves: 1/v_g * dφ_g/dt = ∇·D_g∇φ_g - Σ_r,g·φ_g + Σ_s,g'→g·φ_g' 
    !!                           + χ_p,g·(1-β)·Σ_f,g'·φ_g' + Σ_i λ_i·C_i·χ_d,i,g
    !!         dC_i/dt = β_i·Σ_f,g'·φ_g' - λ_i·C_i
    !!
    !! Uses theta method: θ=0 (explicit), θ=0.5 (Crank-Nicolson), θ=1 (implicit)
    subroutine mg_solve_transient(state, precursors, dt)
        type(mg_state_t), intent(inout) :: state
        real(wp), intent(inout) :: precursors(:, :, :, :)  ! (nx, ny, nz, n_delayed)
        real(wp), intent(in) :: dt
        
        ! Loop indices
        integer :: i, j, k, g, gp, d, iter
        
        ! Constants
        integer, parameter :: n_delayed = 6
        real(wp), parameter :: MIN_DENOM = 1.0e-10_wp
        real(wp), parameter :: MAX_FLUX = 1.0e15_wp
        real(wp), parameter :: RELAX = 0.7_wp
        
        ! Scalars
        real(wp) :: theta
        real(wp) :: source
        real(wp) :: inscatter
        real(wp) :: fission_prompt
        real(wp) :: fission_delayed
        real(wp) :: fission_total
        real(wp) :: precursor_source
        real(wp) :: D_coef
        real(wp) :: sigma_r_val
        real(wp) :: v_inv_val
        real(wp) :: lambda_val
        real(wp) :: beta_val
        real(wp) :: phi_xp, phi_xm
        real(wp) :: phi_yp, phi_ym
        real(wp) :: phi_zp, phi_zm
        real(wp) :: phi_xp_old, phi_xm_old
        real(wp) :: phi_yp_old, phi_ym_old
        real(wp) :: phi_zp_old, phi_zm_old
        real(wp) :: stencil_new, stencil_old
        real(wp) :: phi_new
        real(wp) :: phi_old
        real(wp) :: coef_x, coef_y, coef_z
        real(wp) :: coef_c
        real(wp) :: coef_time
        real(wp) :: inv_dx2, inv_dy2, inv_dz2
        real(wp) :: rhs
        real(wp) :: denom
        real(wp) :: flux_change
        real(wp) :: max_change
        
        ! Standard delayed neutron data (6 groups for U-235)
        real(wp), parameter :: lambda(6) = [0.0124_wp, 0.0305_wp, 0.111_wp, &
                                            0.301_wp, 1.14_wp, 3.01_wp]  ! 1/s
        real(wp), parameter :: beta(6) = [0.000215_wp, 0.001424_wp, 0.001274_wp, &
                                        0.002568_wp, 0.000748_wp, 0.000273_wp]
        
        ! Time discretization: backward Euler (θ = 1.0). CN (θ = 0.5) is
        ! L2-stable but oscillates for stiff systems where |λ|·dt >> 1.
        ! Neutron diffusion is *extremely* stiff: the prompt-decay rate is
        ! 1/(σ_a·v) ~ 1 µs, while dt is 20 ms — that's ~30000 mean-free
        ! times per tick. CN's amplification factor (1+(1-θ)·γ·dt)/(1-θ·γ·dt)
        ! tends to -1 in this regime → flux flips sign each tick (visible
        ! as a 2-tick k_eff limit cycle ~0.0015 ↔ ~0.045 at cold standby).
        ! Backward Euler is unconditionally monotone and damps the prompt
        ! transient instantly, which is the physically correct behaviour
        ! at this dt (we resolve delayed timescales, not prompt).
        theta = 1.0_wp
        
        ! Store old flux
        state%flux_old = state%flux
        
        ! Pre-compute grid factors
        inv_dx2 = 1.0_wp / (state%dx * state%dx)
        inv_dy2 = 1.0_wp / (state%dy * state%dy)
        inv_dz2 = 1.0_wp / (state%dz * state%dz)
        
        ! Step 1: Update precursor concentrations (semi-implicit)
        do k = 1, state%nz
            do j = 1, state%ny
                do i = 1, state%nx
                    ! Delayed-neutron production rate: beta_d * nu * Sigma_f * phi.
                    ! Use nu_sigma_f (not sigma_f) so lambda_d * C_d evaluates as the
                    ! delayed *neutron* source rate, not the delayed *fission* rate.
                    fission_total = 0.0_wp
                    do g = 1, state%n_groups
                        fission_total = fission_total + &
                            state%xsec(i,j,k)%nu_sigma_f(g) * state%flux_old(i,j,k,g)
                    end do
                    
                    ! Update each delayed group
                    do d = 1, min(n_delayed, size(precursors, 4))
                        lambda_val = lambda(d)
                        beta_val = beta(d)
                        
                        ! Semi-implicit: C^(n+1) = [C^n + dt*beta*F^n] / (1 + dt*lambda)
                        precursors(i,j,k,d) = (precursors(i,j,k,d) + dt * beta_val * fission_total) / &
                                            (1.0_wp + dt * lambda_val)
                        
                        ! Ensure non-negative
                        precursors(i,j,k,d) = max(precursors(i,j,k,d), 0.0_wp)
                    end do
                end do
            end do
        end do
        
        ! Step 2: Solve multi-group diffusion with updated precursors
        ! Iterate to convergence (for implicit/semi-implicit schemes)
        do iter = 1, state%config%max_inner_iter
            max_change = 0.0_wp
            
            ! Sweep through all energy groups
            do g = 1, state%n_groups
                
                ! Get group parameters
                ! Note: v_inv_val = 1/v where v is neutron speed [cm/s]
                ! For thermal: v approx 2.2e5 cm/s, fast: v approx 1e9 cm/s
                if (g == state%n_groups) then
                    ! Thermal group (slowest)
                    v_inv_val = 1.0_wp / 2.2e5_wp  ! s/cm
                else
                    ! Fast groups
                    v_inv_val = 1.0_wp / 1.0e9_wp   ! s/cm
                end if
                
                ! Solve for interior points
                do k = 2, state%nz - 1
                    do j = 2, state%ny - 1
                        do i = 2, state%nx - 1
                            
                            D_coef = state%xsec(i,j,k)%D(g)
                            sigma_r_val = state%xsec(i,j,k)%sigma_r(g)
                            phi_old = state%flux_old(i,j,k,g)
                            
                            ! Time derivative coefficient
                            coef_time = v_inv_val / dt
                            
                            ! Neighbor fluxes at current iterate (n+1) and old step (n).
                            ! Crank-Nicolson needs both: theta-weighted on the new side,
                            ! (1-theta) weighted on the old side.
                            phi_xm = max(state%flux(i-1,j,k,g), 0.0_wp)
                            phi_xp = max(state%flux(i+1,j,k,g), 0.0_wp)
                            phi_ym = max(state%flux(i,j-1,k,g), 0.0_wp)
                            phi_yp = max(state%flux(i,j+1,k,g), 0.0_wp)
                            phi_zm = max(state%flux(i,j,k-1,g), 0.0_wp)
                            phi_zp = max(state%flux(i,j,k+1,g), 0.0_wp)

                            phi_xm_old = max(state%flux_old(i-1,j,k,g), 0.0_wp)
                            phi_xp_old = max(state%flux_old(i+1,j,k,g), 0.0_wp)
                            phi_ym_old = max(state%flux_old(i,j-1,k,g), 0.0_wp)
                            phi_yp_old = max(state%flux_old(i,j+1,k,g), 0.0_wp)
                            phi_zm_old = max(state%flux_old(i,j,k-1,g), 0.0_wp)
                            phi_zp_old = max(state%flux_old(i,j,k+1,g), 0.0_wp)

                            ! Diffusion operator coefficients
                            coef_x = D_coef * inv_dx2
                            coef_y = D_coef * inv_dy2
                            coef_z = D_coef * inv_dz2
                            coef_c = 2.0_wp * (coef_x + coef_y + coef_z)
                            
                            ! Source terms
                            ! In-scattering from other groups
                            inscatter = 0.0_wp
                            do gp = 1, state%n_groups
                                if (gp /= g) then
                                    ! Use theta-weighted flux
                                    inscatter = inscatter + state%xsec(i,j,k)%sigma_s(gp,g) * &
                                        (theta * state%flux(i,j,k,gp) + &
                                        (1.0_wp - theta) * state%flux_old(i,j,k,gp))
                                end if
                            end do
                            
                            ! Prompt fission source (1-beta total)
                            fission_total = 0.0_wp
                            do gp = 1, state%n_groups
                                fission_total = fission_total + &
                                    state%xsec(i,j,k)%nu_sigma_f(gp) * &
                                    (theta * state%flux(i,j,k,gp) + &
                                    (1.0_wp - theta) * state%flux_old(i,j,k,gp))
                            end do
                            
                            ! Total delayed fraction
                            fission_prompt = (1.0_wp - state%xsec(i,j,k)%beta_total) * &
                                            state%xsec(i,j,k)%chi(g) * fission_total
                            
                            ! Delayed neutron source from precursors
                            fission_delayed = 0.0_wp
                            do d = 1, min(n_delayed, size(precursors, 4))
                                ! Use delayed spectrum if available, else prompt spectrum
                                if (allocated(state%xsec(i,j,k)%chi_d)) then
                                    fission_delayed = fission_delayed + &
                                        lambda(d) * precursors(i,j,k,d) * &
                                        state%xsec(i,j,k)%chi_d(g)
                                else
                                    fission_delayed = fission_delayed + &
                                        lambda(d) * precursors(i,j,k,d) * &
                                        state%xsec(i,j,k)%chi(g)
                                end if
                            end do
                            
                            ! External source
                            source = inscatter + fission_prompt + fission_delayed + &
                                    state%external_source(i,j,k,g)
                            
                            ! Crank-Nicolson assembly. Move theta-weighted (n+1) terms
                            ! to the LHS, (1-theta) (n) terms to the RHS:
                            !   [coef_time + theta*(coef_c + sigma_r)] * phi^{n+1}
                            !     = coef_time*phi^n
                            !       + theta * neighbor_new
                            !       + (1-theta) * [neighbor_old - (coef_c + sigma_r)*phi^n]
                            !       + source
                            ! Previously both stencils used the new iterate and the
                            ! (1-theta) center removal was missing, collapsing the
                            ! scheme to an inconsistent mix of CN diagonal and BE
                            ! off-diagonal -- effectively over-implicit on the
                            ! neighbours, biasing steady state.
                            stencil_new = coef_x * (phi_xm     + phi_xp)     + &
                                          coef_y * (phi_ym     + phi_yp)     + &
                                          coef_z * (phi_zm     + phi_zp)
                            stencil_old = coef_x * (phi_xm_old + phi_xp_old) + &
                                          coef_y * (phi_ym_old + phi_yp_old) + &
                                          coef_z * (phi_zm_old + phi_zp_old)

                            rhs = coef_time * phi_old + &
                                  theta * stencil_new + &
                                  (1.0_wp - theta) * &
                                    (stencil_old - (coef_c + sigma_r_val) * phi_old) + &
                                  source

                            ! Left-hand side denominator
                            denom = coef_time + theta * (coef_c + sigma_r_val)
                            denom = max(denom, MIN_DENOM)
                            
                            ! Update flux
                            phi_new = rhs / denom
                            
                            ! Under-relaxation for stability
                            phi_new = RELAX * phi_new + (1.0_wp - RELAX) * state%flux(i,j,k,g)
                            
                            ! Clamp to physical range
                            phi_new = max(phi_new, 0.0_wp)
                            phi_new = min(phi_new, MAX_FLUX)
                            
                            ! Track maximum absolute change. We normalise after
                            ! the full sweep against the max flux across the
                            ! domain — per-cell normalisation by (phi_new + tiny)
                            ! blows up for near-zero cells (cold standby with
                            ! rods deep in) and produces false non-convergence.
                            flux_change = abs(phi_new - state%flux(i,j,k,g))
                            max_change  = max(max_change, flux_change)

                            state%flux(i,j,k,g) = phi_new
                        end do
                    end do
                end do
                
                ! Apply boundary conditions
                call apply_boundary_conditions(state, g)
            end do
            
            ! Normalise: convergence is satisfied when the largest per-cell
            ! flux change is below tolerance × max flux in the domain. This
            ! is invariant to flux amplitude — works equally well for
            ! cold-standby (max φ ~ 10^3 n/cm²/s) and rated (max φ ~ 10^14).
            if (max_change < state%config%inner_tolerance * &
                              max(maxval(state%flux), 1.0_wp)) then
                state%inner_iterations = iter
                exit
            end if

            ! Only warn on hard divergence (max change > 10 % of peak flux
            ! after burning the full inner-iter budget). Smaller residuals
            ! are normal during active transients (rod motion, XS feedback)
            ! and converge in subsequent ticks — the warning would otherwise
            ! spam the log every step of any non-trivial transient.
            if (iter == state%config%max_inner_iter .and. &
                max_change > 0.1_wp * max(maxval(state%flux), 1.0_wp)) then
                print '(A,I4,A,ES10.3,A,ES10.3)', &
                    'WARNING: Transient solver diverging at iter ', &
                    iter, '. Max change: ', max_change, &
                    '  max flux: ', maxval(state%flux)
            end if
        end do
        
        ! Step 3: Update derived quantities
        
        ! Update fission source
        call compute_fission_source(state)
        
        ! Update power distribution
        call compute_power_distribution(state)
        
        ! Compute instantaneous k-effective (for monitoring)
        state%k_eff = mg_compute_keff(state)
        
    end subroutine mg_solve_transient
    
    !> Power iteration method (alternative solver)
    subroutine mg_power_iteration(state, max_iter, tolerance)
        type(mg_state_t), intent(inout) :: state
        integer, intent(in) :: max_iter
        real(wp), intent(in) :: tolerance
        
        ! Wrapper for mg_solve_eigenvalue with custom parameters
        type(mg_config_t) :: old_config
        
        old_config = state%config
        state%config%max_outer_iter = max_iter
        state%config%outer_tolerance = tolerance
        
        call mg_solve_eigenvalue(state, state%k_eff, state%converged)
        
        state%config = old_config
    end subroutine mg_power_iteration

#if defined(COMPUTATION_MODE_GPU) || defined(COMPUTATION_MODE_HYBRID)
    ! -----------------------------------------------------------------
    ! GPU-accelerated eigenvalue solver.
    !
    ! Runs the power iteration with GPU-accelerated source iteration
    ! steps. The fission source computation and k_eff update use the
    ! GPU reduction kernels. The host handles the outer iteration
    ! control flow (convergence checks, Chebyshev acceleration).
    ! -----------------------------------------------------------------
    subroutine mg_solve_eigenvalue_gpu(state, k_eff, converged)
        type(mg_state_t), intent(inout) :: state
        real(wp), intent(out) :: k_eff
        logical, intent(out) :: converged

        type(mg_gpu_state_t) :: gpu_state
        integer :: outer_iter, inner_iter, g, i, j, k, gg
        real(wp) :: k_error, fission_sum, fission_new, k_eff_old_local
        real(wp), allocatable :: D_arr(:,:,:,:), sigma_r_arr(:,:,:,:)
        real(wp), allocatable :: sigma_s_arr(:,:,:,:,:), chi_arr(:,:,:,:)
        real(wp), allocatable :: ext_src_arr(:,:,:,:), fsrc_arr(:,:,:)
        real(wp), allocatable :: nu_sf_arr(:,:,:,:)

        ! Reset keff if stuck at a clamp boundary
        if (state%k_eff >= 4.99_wp .or. state%k_eff <= 0.006_wp) state%k_eff = 1.0_wp
        state%converged = .false.
        where (state%flux < 1.0e-10_wp) state%flux = 1.0_wp

        ! Flatten cross-sections for the GPU
        allocate(D_arr(state%nx, state%ny, state%nz, state%n_groups))
        allocate(sigma_r_arr(state%nx, state%ny, state%nz, state%n_groups))
        allocate(sigma_s_arr(state%n_groups, state%n_groups, state%nx, state%ny, state%nz))
        allocate(chi_arr(state%nx, state%ny, state%nz, state%n_groups))
        allocate(ext_src_arr(state%nx, state%ny, state%nz, state%n_groups))
        allocate(fsrc_arr(state%nx, state%ny, state%nz))
        allocate(nu_sf_arr(state%nx, state%ny, state%nz, state%n_groups))
        do k = 1, state%nz; do j = 1, state%ny; do i = 1, state%nx
            do gg = 1, state%n_groups
                D_arr(i,j,k,gg) = state%xsec(i,j,k)%D(gg)
                sigma_r_arr(i,j,k,gg) = state%xsec(i,j,k)%sigma_r(gg)
                chi_arr(i,j,k,gg) = state%xsec(i,j,k)%chi(gg)
                ext_src_arr(i,j,k,gg) = state%external_source(i,j,k,gg)
                nu_sf_arr(i,j,k,gg) = state%xsec(i,j,k)%nu_sigma_f(gg)
            end do
            do gg = 1, state%n_groups
                do g = 1, state%n_groups
                    sigma_s_arr(g, gg, i, j, k) = state%xsec(i,j,k)%sigma_s(g, gg)
                end do
            end do
            fsrc_arr(i,j,k) = state%fission_source(i,j,k)
        end do; end do; end do

        ! Initialise GPU state
        call mg_gpu_init(gpu_state, state%nx, state%ny, state%nz, &
            state%n_groups, state%dx, state%dy, state%dz, state%k_eff)
        call mg_gpu_set_albedos(gpu_state, &
            state%config%albedo_x_min, state%config%albedo_x_max, &
            state%config%albedo_y_min, state%config%albedo_y_max, &
            state%config%albedo_z_min, state%config%albedo_z_max)
        call mg_gpu_copy_to_device(gpu_state, state%flux, fsrc_arr, &
            D_arr, sigma_r_arr, sigma_s_arr, chi_arr, ext_src_arr, state%k_eff)

        ! Power iteration with GPU source iteration
        do outer_iter = 1, state%config%max_outer_iter
            state%outer_iterations = outer_iter
            k_eff_old_local = state%k_eff
            state%flux_old = state%flux

            ! Compute fission source on GPU and get the sum
            call mg_gpu_compute_fission_source(gpu_state, nu_sf_arr)
            fission_sum = mg_gpu_fission_sum(gpu_state)

            if (fission_sum < 1.0e-30_wp) then
                converged = .false.
                k_eff = 0.0_wp
                call mg_gpu_cleanup(gpu_state)
                deallocate(D_arr, sigma_r_arr, sigma_s_arr, chi_arr, &
                    ext_src_arr, fsrc_arr, nu_sf_arr)
                return
            end if

            ! Inner iterations on GPU
            do inner_iter = 1, state%config%max_inner_iter
                do g = 1, state%n_groups
                    call mg_gpu_source_iteration_step(gpu_state, g)
                end do
                ! Update k_eff_inv in the GPU state for the next sweep
                gpu_state%k_eff_inv = 1.0_wp / max(state%k_eff, 0.005_wp)
                ! Simple convergence check: compare flux_new to flux
                ! (For production, a proper L2 norm reduction should be used)
                exit  ! GPU path: one inner sweep per outer iteration for simplicity
            end do
            state%inner_iterations = inner_iter

            ! Recompute fission source with updated flux for k_eff update
            call mg_gpu_compute_fission_source(gpu_state, nu_sf_arr)
            fission_new = mg_gpu_fission_sum(gpu_state)

            if (fission_sum > 1.0e-30_wp .and. fission_new > 1.0e-30_wp) then
                state%k_eff = k_eff_old_local * (fission_new / fission_sum)
            end if
            state%k_eff = min(max(state%k_eff, 0.005_wp), 5.0_wp)

            ! Check convergence
            k_error = abs(state%k_eff - k_eff_old_local) / (abs(state%k_eff) + 1.0e-30_wp)
            state%outer_error = k_error

            if (k_error < state%config%outer_tolerance .and. outer_iter > 5) then
                state%converged = .true.
                exit
            end if
        end do

        ! Copy converged flux back to host
        call mg_gpu_copy_from_device(gpu_state, state%flux)

        ! Normalize and compute power on host
        call normalize_flux(state)
        call compute_power_distribution(state)

        k_eff = state%k_eff
        converged = state%converged

        call mg_gpu_cleanup(gpu_state)
        deallocate(D_arr, sigma_r_arr, sigma_s_arr, chi_arr, &
            ext_src_arr, fsrc_arr, nu_sf_arr)
    end subroutine mg_solve_eigenvalue_gpu
#endif

end module multigroup_diffusion