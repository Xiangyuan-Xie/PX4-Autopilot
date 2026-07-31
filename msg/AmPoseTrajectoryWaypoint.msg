# AM Pose trajectory waypoint in the local NED frame.
# Array index i is sampled at (i + 1) * 0.1 seconds after acceptance.

uint64 timestamp # [us] Parent command timestamp; not used for trajectory timing

float32[3] position # [m] [@frame NED]
float32[3] velocity # [m/s] [@frame NED]
float32[3] acceleration # [m/s^2] [@frame NED]

float32 yaw # [rad] [@frame NED]
float32 yaw_rate # [rad/s] [@frame NED]
float32 yaw_acceleration # [rad/s^2] [@frame NED]
