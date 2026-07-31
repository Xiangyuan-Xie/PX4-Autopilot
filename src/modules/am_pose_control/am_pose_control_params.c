/****************************************************************************
 *
 * Parameters for the AM Pose module.
 *
 ****************************************************************************/

#include <parameters/param.h>

/**
 * Maximum horizontal forward manual velocity for AM Pose.
 *
 * @unit m/s
 * @min 0
 * @max 20
 * @increment 1
 * @decimal 1
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_VEL_MANUAL, 1.5f);

/**
 * Maximum horizontal sideways manual velocity for AM Pose.
 *
 * Default is `-1`, which means "follow AMPC_VEL_MANUAL".
 *
 * @unit m/s
 * @min -1
 * @max 20
 * @increment 1
 * @decimal 1
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_VEL_SIDE, -1.f);

/**
 * Maximum horizontal backward manual velocity for AM Pose.
 *
 * Default is `-1`, which means "follow AMPC_VEL_MANUAL".
 *
 * @unit m/s
 * @min -1
 * @max 20
 * @increment 1
 * @decimal 1
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_VEL_BACK, -1.f);

/**
 * Maximum upward manual velocity for AM Pose.
 *
 * @unit m/s
 * @min 0
 * @max 8
 * @decimal 1
 * @increment 0.1
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_Z_VEL_UP, 1.0f);

/**
 * Maximum downward manual velocity for AM Pose.
 *
 * @unit m/s
 * @min 0
 * @max 4
 * @decimal 1
 * @increment 0.1
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_Z_VEL_DN, 1.0f);

/**
 * Takeoff ramp time for AM Pose.
 *
 * This only controls the AM takeoff state machine and output release gate. It
 * does not ramp policy observations or scale policy motor outputs.
 *
 * @unit s
 * @min 0
 * @max 10
 * @decimal 2
 * @increment 0.1
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_TKO_RAMP_T, 3.f);

/**
 * Trajectory setpoint freshness timeout for AM Pose.
 *
 * Position targets are held after this timeout. Velocity targets are replaced
 * by zero velocity and yaw rate, causing a jerk-limited stop and pose hold.
 *
 * @unit s
 * @min 0
 * @max 5
 * @decimal 2
 * @increment 0.1
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_TRJ_TIMEOUT, 0.5f);

/**
 * Trajectory setpoint loss timeout for AM Pose.
 *
 * After this timeout AM clears a latched velocity command and records the input
 * as lost. The local reference continues its jerk-limited stop and pose hold.
 *
 * @unit s
 * @min 0
 * @max 10
 * @decimal 2
 * @increment 0.1
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_TRJ_LOSS_T, 1.0f);

/**
 * Maximum AM Pose Offboard position and velocity reference speed.
 *
 * This bounds position and velocity commands consumed by AM Pose Offboard. Manual AM Pose
 * speed is configured independently by AMPC_VEL_MANUAL, AMPC_VEL_SIDE and
 * AMPC_VEL_BACK.
 *
 * @unit m/s
 * @min 0.1
 * @max 1.0
 * @decimal 2
 * @increment 0.1
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_POSE_VMAX, 1.0f);

/**
 * Maximum AM Pose Offboard reach-and-hold and velocity reference yaw rate.
 *
 * @unit deg/s
 * @min 1
 * @max 57.3
 * @decimal 1
 * @increment 1
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_POSE_YMAX, 57.3f);

/**
 * Blend time after AM Pose mode changes or estimator resets.
 *
 * @unit s
 * @min 0
 * @max 0.5
 * @decimal 2
 * @increment 0.01
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_SW_BLEND_T, 0.10f);

/**
 * Maximum manual yaw rate for AM Pose.
 *
 * @unit deg/s
 * @min 0
 * @max 400
 * @decimal 0
 * @increment 10
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_MAN_Y_MAX, 57.3f);

/**
 * Manual yaw rate filter time constant for AM Pose.
 *
 * @unit s
 * @min 0
 * @max 5
 * @decimal 2
 * @increment 0.01
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_MAN_Y_TAU, 0.08f);

/**
 * Maximum horizontal manual reference acceleration for AM Pose.
 *
 * @unit m/s^2
 * @min 0.1
 * @max 20
 * @decimal 2
 * @increment 0.1
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_MAN_ACC_HOR, 3.0f);

/**
 * Maximum horizontal manual reference jerk for AM Pose.
 *
 * @unit m/s^3
 * @min 0.1
 * @max 100
 * @decimal 2
 * @increment 0.1
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_MAN_JRK_HOR, 8.0f);

/**
 * Maximum upward manual reference acceleration for AM Pose.
 *
 * @unit m/s^2
 * @min 0.1
 * @max 20
 * @decimal 2
 * @increment 0.1
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_MAN_ACC_UP, 3.0f);

/**
 * Maximum downward manual reference acceleration for AM Pose.
 *
 * @unit m/s^2
 * @min 0.1
 * @max 20
 * @decimal 2
 * @increment 0.1
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_MAN_ACC_DN, 3.0f);

/**
 * Maximum vertical manual reference jerk for AM Pose.
 *
 * @unit m/s^3
 * @min 0.1
 * @max 100
 * @decimal 2
 * @increment 0.1
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_MAN_JERK_Z, 8.0f);

/**
 * Maximum manual yaw reference acceleration for AM Pose.
 *
 * @unit rad/s^2
 * @min 0.1
 * @max 50
 * @decimal 2
 * @increment 0.1
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_MAN_Y_ACC, 3.0f);

/**
 * Maximum manual yaw reference jerk for AM Pose.
 *
 * @unit rad/s^3
 * @min 0.1
 * @max 100
 * @decimal 2
 * @increment 0.1
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_MAN_Y_JERK, 8.0f);

/**
 * Manual stick deadzone for AM Pose.
 *
 * @unit norm
 * @min 0
 * @max 0.4
 * @decimal 2
 * @increment 0.01
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_MAN_DZ, 0.1f);

/**
 * Maximum vertical velocity for engaging hold in AM Pose.
 *
 * @unit m/s
 * @min 0
 * @max 3
 * @decimal 2
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_HOLD_MAX_Z, 0.6f);

/**
 * Maximum horizontal velocity for engaging hold in AM Pose.
 *
 * @unit m/s
 * @min 0
 * @max 3
 * @decimal 2
 * @group AM Pose
 */
PARAM_DEFINE_FLOAT(AMPC_HOLD_MAX_XY, 0.8f);
