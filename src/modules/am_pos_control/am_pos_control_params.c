/****************************************************************************
 *
 * Parameters for the AM Position module.
 *
 ****************************************************************************/

#include <parameters/param.h>

/**
 * Maximum horizontal forward manual velocity for AM Position.
 *
 * @unit m/s
 * @min 0
 * @max 20
 * @increment 1
 * @decimal 1
 * @group AM Position
 */
PARAM_DEFINE_FLOAT(AMPC_VEL_MANUAL, 1.5f);

/**
 * Maximum horizontal sideways manual velocity for AM Position.
 *
 * Default is `-1`, which means "follow AMPC_VEL_MANUAL".
 *
 * @unit m/s
 * @min -1
 * @max 20
 * @increment 1
 * @decimal 1
 * @group AM Position
 */
PARAM_DEFINE_FLOAT(AMPC_VEL_SIDE, -1.f);

/**
 * Maximum horizontal backward manual velocity for AM Position.
 *
 * Default is `-1`, which means "follow AMPC_VEL_MANUAL".
 *
 * @unit m/s
 * @min -1
 * @max 20
 * @increment 1
 * @decimal 1
 * @group AM Position
 */
PARAM_DEFINE_FLOAT(AMPC_VEL_BACK, -1.f);

/**
 * Maximum upward manual velocity for AM Position.
 *
 * @unit m/s
 * @min 0
 * @max 8
 * @decimal 1
 * @increment 0.1
 * @group AM Position
 */
PARAM_DEFINE_FLOAT(AMPC_Z_VEL_UP, 1.0f);

/**
 * Maximum downward manual velocity for AM Position.
 *
 * @unit m/s
 * @min 0
 * @max 4
 * @decimal 1
 * @increment 0.1
 * @group AM Position
 */
PARAM_DEFINE_FLOAT(AMPC_Z_VEL_DN, 1.0f);

/**
 * Maximum manual yaw rate for AM Position.
 *
 * @unit deg/s
 * @min 0
 * @max 400
 * @decimal 0
 * @increment 10
 * @group AM Position
 */
PARAM_DEFINE_FLOAT(AMPC_MAN_Y_MAX, 57.3f);

/**
 * Manual yaw rate filter time constant for AM Position.
 *
 * @unit s
 * @min 0
 * @max 5
 * @decimal 2
 * @increment 0.01
 * @group AM Position
 */
PARAM_DEFINE_FLOAT(AMPC_MAN_Y_TAU, 0.08f);

/**
 * Manual stick deadzone for AM Position.
 *
 * @unit norm
 * @min 0
 * @max 0.4
 * @decimal 2
 * @increment 0.01
 * @group AM Position
 */
PARAM_DEFINE_FLOAT(AMPC_MAN_DZ, 0.1f);

/**
 * Maximum vertical velocity for engaging hold in AM Position.
 *
 * @unit m/s
 * @min 0
 * @max 3
 * @decimal 2
 * @group AM Position
 */
PARAM_DEFINE_FLOAT(AMPC_HOLD_MAX_Z, 0.6f);

/**
 * Maximum horizontal velocity for engaging hold in AM Position.
 *
 * @unit m/s
 * @min 0
 * @max 3
 * @decimal 2
 * @group AM Position
 */
PARAM_DEFINE_FLOAT(AMPC_HOLD_MAX_XY, 0.8f);
