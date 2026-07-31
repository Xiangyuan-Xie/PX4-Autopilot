# Off-board control mode

uint64 timestamp		# time since system start (microseconds)

bool position
bool velocity
bool acceleration
bool attitude
bool body_rate
bool thrust_and_torque
bool direct_actuator

uint8 CONTROLLER_TYPE_NATIVE = 0
uint8 CONTROLLER_TYPE_AM_POSE = 1
uint8 controller_type
