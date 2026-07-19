uint64 timestamp

bool enabled
bool config_valid
uint8 matrix_rank
bool matrix_full_rank
bool hover_feasible
bool allocation_failure

float32 condition_number
float32[6] normalization_scale             # physical wrench to normalized control scale
float32 hover_residual                     # maximum normalized wrench residual
float32 hover_actuator_min
float32 hover_actuator_max

float32[6] requested_wrench                # normalized [Mx, My, Mz, Fx, Fy, Fz]
float32[6] applied_wrench                  # normalized [Mx, My, Mz, Fx, Fy, Fz]
float32[6] unallocated_wrench              # requested - applied, normalized
float32 unallocated_wrench_norm            # Euclidean norm of normalized residual
