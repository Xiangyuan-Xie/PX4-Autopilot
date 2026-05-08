uint64 timestamp
uint64 timestamp_sample

bool am_valid
uint32 failure_flags
uint32 degraded_flags

uint32 FAILURE_NONE = 0
uint32 FAILURE_VEHICLE_STATE_INVALID = 1
uint32 FAILURE_ARM_STATE_INVALID = 2
uint32 FAILURE_AM_SETPOINT_INVALID = 4
uint32 FAILURE_AM_INFERENCE_FAILED = 8

uint32 DEGRADED_NONE = 0
uint32 DEGRADED_LOCAL_XY_INVALID = 1
uint32 DEGRADED_LOCAL_VXY_INVALID = 2
uint32 DEGRADED_SETPOINT_DEFAULTED = 4

float32[4] am_raw_action
float32[4] am_mapped_action

float32[12] am_motor_control

uint64 am_setpoint_timestamp

# TOPICS am_test_result
