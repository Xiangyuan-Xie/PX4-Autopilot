uint64 timestamp

bool module_running
bool vehicle_state_valid
bool arm_state_valid
bool am_setpoint_valid
bool am_valid
uint32 failure_flags
uint32 degraded_flags

uint32 DEGRADED_NONE = 0
uint32 DEGRADED_LOCAL_XY_INVALID = 1
uint32 DEGRADED_LOCAL_VXY_INVALID = 2
uint32 DEGRADED_SETPOINT_DEFAULTED = 4

# TOPICS am_test_status
