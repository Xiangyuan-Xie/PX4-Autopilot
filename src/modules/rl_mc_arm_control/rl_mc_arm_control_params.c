/****************************************************************************
 *
 * Parameters for the RL multicopter arm control module.
 *
 ****************************************************************************/

#include <parameters/param.h>

/**
 * If true the RL arm control module is automatically started on boot.
 *
 * @boolean
 * @group RL Arm Control
 */
PARAM_DEFINE_INT32(RL_ARM_EN, 1);

/**
 * Enable trajectory generation from manual control sticks.
 *
 * @boolean
 * @reboot_required true
 * @group RL Arm Control
 */
PARAM_DEFINE_INT32(RL_ARM_MANL_CTRL, 0);

/**
 * Maximum horizontal forward manual velocity for RL arm control mode.
 *
 * Default matches the training command generator range
 * `CommandsCfg.base_velocity.ranges.lin_vel_x` in `mc_arm_position/env_cfg.py`.
 *
 * @unit m/s
 * @min 0
 * @max 20
 * @increment 1
 * @decimal 1
 * @group RL Arm Control
 */
PARAM_DEFINE_FLOAT(MAPC_VEL_MANUAL, 1.5f);

/**
 * Maximum horizontal sideways manual velocity for RL arm control mode.
 *
 * Default is `-1`, which means "follow MAPC_VEL_MANUAL".
 * This keeps the effective default aligned with the symmetric training
 * command generator range `CommandsCfg.base_velocity.ranges.lin_vel_y`.
 *
 * @unit m/s
 * @min -1
 * @max 20
 * @increment 1
 * @decimal 1
 * @group RL Arm Control
 */
PARAM_DEFINE_FLOAT(MAPC_VEL_SIDE, -1.f);

/**
 * Maximum horizontal backward manual velocity for RL arm control mode.
 *
 * Default is `-1`, which means "follow MAPC_VEL_MANUAL".
 * This keeps the effective default aligned with the training command
 * generator's symmetric forward/backward linear velocity range.
 *
 * @unit m/s
 * @min -1
 * @max 20
 * @increment 1
 * @decimal 1
 * @group RL Arm Control
 */
PARAM_DEFINE_FLOAT(MAPC_VEL_BACK, -1.f);

/**
 * Maximum horizontal acceleration for RL arm control mode.
 *
 * Default matches MPC_ACC_HOR.
 *
 * @unit m/s^2
 * @min 0
 * @max 15
 * @decimal 2
 * @increment 1
 * @group RL Arm Control
 */
PARAM_DEFINE_FLOAT(MAPC_ACC_HOR, 3.f);

/**
 * Maximum horizontal and vertical jerk for RL arm control mode.
 *
 * Default matches MPC_JERK_MAX.
 *
 * @unit m/s^3
 * @min 0.5
 * @max 500
 * @decimal 0
 * @increment 1
 * @group RL Arm Control
 */
PARAM_DEFINE_FLOAT(MAPC_JERK_MAX, 8.f);

/**
 * Maximum upward manual velocity for RL arm control mode.
 *
 * Default matches the training command generator range
 * `CommandsCfg.base_velocity.ranges.lin_vel_z` in `mc_arm_position/env_cfg.py`.
 *
 * @unit m/s
 * @min 0
 * @max 8
 * @decimal 1
 * @increment 0.1
 * @group RL Arm Control
 */
PARAM_DEFINE_FLOAT(MAPC_Z_VEL_UP, 1.0f);

/**
 * Maximum downward manual velocity for RL arm control mode.
 *
 * Default matches the training command generator range
 * `CommandsCfg.base_velocity.ranges.lin_vel_z` in `mc_arm_position/env_cfg.py`.
 *
 * @unit m/s
 * @min 0
 * @max 4
 * @decimal 1
 * @increment 0.1
 * @group RL Arm Control
 */
PARAM_DEFINE_FLOAT(MAPC_Z_VEL_DN, 1.0f);

/**
 * Maximum manual yaw rate for RL arm control mode.
 *
 * Default matches the training command generator range
 * `CommandsCfg.base_velocity.ranges.ang_vel_z = +/- 1.0 rad/s`
 * in `mc_arm_position/env_cfg.py`.
 *
 * @unit deg/s
 * @min 0
 * @max 400
 * @decimal 0
 * @increment 10
 * @group RL Arm Control
 */
PARAM_DEFINE_FLOAT(MAPC_MAN_Y_MAX, 57.3f);

/**
 * Manual yaw rate filter time constant for RL arm control mode.
 *
 * Default matches MPC_MAN_Y_TAU.
 *
 * @unit s
 * @min 0
 * @max 5
 * @decimal 2
 * @increment 0.01
 * @group RL Arm Control
 */
PARAM_DEFINE_FLOAT(MAPC_MAN_Y_TAU, 0.08f);

/**
 * Manual stick deadzone for RL arm control mode.
 *
 * Default matches MAN_DEADZONE.
 *
 * @unit norm
 * @min 0
 * @max 0.4
 * @decimal 2
 * @increment 0.01
 * @group RL Arm Control
 */
PARAM_DEFINE_FLOAT(MAPC_MAN_DZ, 0.1f);

/**
 * Maximum vertical velocity for engaging hold in RL arm control mode.
 *
 * Default matches MPC_HOLD_MAX_Z.
 *
 * @unit m/s
 * @min 0
 * @max 3
 * @decimal 2
 * @group RL Arm Control
 */
PARAM_DEFINE_FLOAT(MAPC_HOLD_MAX_Z, 0.6f);

/**
 * Maximum horizontal velocity for engaging hold in RL arm control mode.
 *
 * Default matches MPC_HOLD_MAX_XY.
 *
 * @unit m/s
 * @min 0
 * @max 3
 * @decimal 2
 * @group RL Arm Control
 */
PARAM_DEFINE_FLOAT(MAPC_HOLD_MAX_XY, 0.8f);
