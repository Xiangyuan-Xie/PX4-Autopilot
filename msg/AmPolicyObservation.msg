uint64 timestamp
uint64 timestamp_sample

uint32 failure_flags
uint32 degraded_flags

uint32 DEGRADED_NONE = 0
uint32 DEGRADED_LOCAL_XY_INVALID = 1
uint32 DEGRADED_LOCAL_VXY_INVALID = 2
uint32 DEGRADED_SETPOINT_DEFAULTED = 4

float32[30] observation
float32[4] raw_action
float32[4] mapped_action

float32[12] motor_control

uint64 am_setpoint_timestamp

# TOPICS am_policy_observation
