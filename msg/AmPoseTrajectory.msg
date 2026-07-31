# Atomic, fixed-capacity trajectory command for the AM Pose Offboard controller.
# Points are spaced by 0.1 seconds; unused array entries are ignored.

uint64 timestamp # [us] Time since system start
uint32 trajectory_id # Change when replacing the active trajectory

uint8 COMMAND_REPLACE = 0
uint8 COMMAND_CANCEL = 1
uint8 command

uint8 MAX_POINTS = 5
uint8 point_count
bool continue_at_end # Permit nonzero final derivatives for a rolling replacement
AmPoseTrajectoryWaypoint[5] points

# TOPICS am_pose_trajectory
