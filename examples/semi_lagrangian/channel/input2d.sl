// physical parameters
PI = 3.14159265359
LX = 4.0
LY = 1.0
RATIO = LX / LY

// grid spacing parameters
MAX_LEVELS = 1                            // maximum number of levels in locally refined grid
REF_RATIO  = 4                            // refinement ratio between levels
NY = 64                                    // coarsest grid spacing
NY_FINEST = (REF_RATIO^(MAX_LEVELS - 1))*NY  // finest   grid spacing
NX = NY * RATIO
NX_FINEST = NY_FINEST * RATIO
H = LX/NX_FINEST
CFL_MAX = 0.5
U_MAX = 15.0
DT_MIN = H * CFL_MAX / 15.0
DT_MAX = DT_MIN
END_TIME = 2.0
REGRID_INTERVAL = 1
TAG_BUFFER = 1

OUTPUT_BDRY_INFO = FALSE
USING_LS_FCN = TRUE
DRAW_EXACT = FALSE
MIN_REFINE_FACTOR = -4.0
MAX_REFINE_FACTOR = 2.0
LEAST_SQUARES_ORDER = "QUADRATIC"
USE_STRANG_SPLITTING = TRUE
USE_OUTSIDE_LS_FOR_TAGGING = TRUE
ADV_INT_METHOD = "MIDPOINT_RULE"
DIF_INT_METHOD = "TRAPEZOIDAL_RULE"
USE_RBFS =  TRUE
RBF_STENCIL_SIZE = 12
RBF_POLY_ORDER = "QUADRATIC"
USE_LAGRANGE = TRUE
USE_SL = TRUE

XCOM = 1.5
YCOM = 0.5
R = 1.0
R_2 = 3.75
R_1 = 0.5
OMEGA_2 = 1.0
OMEGA_1 = 10.0
ROT_PERIOD = 2.0

QInitial {
   type = "TRIANGLE"
   com = XCOM, YCOM
}


UFunction {
   L = LY
   function_0 = "X_1 * (L - X_1)"
   function_1 = "0.0"
}

Q_bcs {
    D = 1.0

    acoef_function_0 = "0.0"
    acoef_function_1 = "0.0"
    acoef_function_2 = "0.0"
    acoef_function_3 = "0.0"

    bcoef_function_0 = "1.0"
    bcoef_function_1 = "1.0"
    bcoef_function_2 = "1.0"
    bcoef_function_3 = "1.0"

    gcoef_function_0 = "0.0"
    gcoef_function_1 = "0.0"
    gcoef_function_2 = "0.0"
    gcoef_function_3 = "0.0"
}

Main {

// log file parameters
   log_file_name               = "test.log"
   log_all_nodes               = FALSE

// visualization dump parameters
   viz_writer                  = "VisIt"
   viz_dump_interval           = int(0.05 / DT_MAX) 
   viz_dump_dirname            = "VIZ.SL"
   visit_number_procs_per_file = 1

// restart dump parameters
   restart_dump_interval       = 0
   restart_dump_dirname        = "restart_adv_diff2d"

   data_dump_interval          = 1//NFINEST/8
   data_dump_dirname           = "hier_data_IB2d"

   timer_dump_interval = 0
}

CartesianGeometry {
   domain_boxes = [ (0,0),(NX - 1,NY - 1) ]
   x_lo = 0.0, 0.0
   x_up = LX, LY
   periodic_dimension = 0,0
}

GriddingAlgorithm {
   max_levels = MAX_LEVELS
   ratio_to_coarser {
      level_1 = REF_RATIO,REF_RATIO
      level_2 = REF_RATIO,REF_RATIO
      level_3 = REF_RATIO,REF_RATIO
   }
   largest_patch_size {
      level_0 = 512,512  // all finer levels will use same values as level_0
   }
   smallest_patch_size {
      level_0 =   4,  4  // all finer levels will use same values as level_0
   }
   efficiency_tolerance = 0.9e0  // min % of tag cells in new patch level
   combine_efficiency   = 0.9e0  // chop box if sum of volumes of smaller boxes < efficiency * vol of large box
}

StandardTagAndInitialize {
   tagging_method = "GRADIENT_DETECTOR"
}

LoadBalancer {
   bin_pack_method     = "SPATIAL"
   max_workload_factor = 1
}

TimerManager{
   print_exclusive = TRUE
   print_total     = TRUE
   print_threshold = 0.0
   timer_list      = "LS::*::*","IBAMR::*::*","IBTK::*::*"
}


LSAdvDiffIntegrator {
    start_time           = 0.0e0  // initial simulation time
    end_time             = END_TIME    // final simulation time
    grow_dt              = 2.0e0  // growth factor for timesteps
    regrid_interval      = REGRID_INTERVAL  // effectively disable regridding
    cfl                  = CFL_MAX
    dt_max               = DT_MAX
    dt_min               = DT_MIN
    enable_logging       = TRUE

    prescribe_level_set  = USING_LS_FCN
    min_ls_refine_factor = MIN_REFINE_FACTOR
    max_ls_refine_factor = MAX_REFINE_FACTOR
    least_squares_order = LEAST_SQUARES_ORDER
    use_strang_splitting = USE_STRANG_SPLITTING
    advection_ts_type = ADV_INT_METHOD
    diffusion_ts_type = DIF_INT_METHOD
    use_rbfs = USE_RBFS
    rbf_stencil_size = RBF_STENCIL_SIZE
    rbf_poly_order = RBF_POLY_ORDER
    default_adv_reconstruct_type = "ZSPLINES"

    tag_buffer = TAG_BUFFER
    
    convective_op_type                 = "PPM"//WAVE_PROP"
    convective_difference_form         = "CONSERVATIVE"
    convective_time_stepping_type      = "TRAPEZOIDAL_RULE"
    init_convective_time_stepping_type = "MIDPOINT_RULE"
    diffusion_time_stepping_type = "TRAPEZOIDAL_RULE"
    num_cycles = 1
}
