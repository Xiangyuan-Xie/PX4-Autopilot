/****************************************************************************
 *
 * Flight task for AM Position manual setpoint generation.
 *
 ****************************************************************************/

#include "FlightTaskManualAmPosition.hpp"

#include <float.h>
#include <geo/geo.h>
#include <mathlib/mathlib.h>

using namespace matrix;

constexpr float FlightTaskManualAmPosition::kExpo;
constexpr float FlightTaskManualAmPosition::kYawErrorTimeConstant;
constexpr float FlightTaskManualAmPosition::kYawErrorChangeThreshold;

bool FlightTaskManualAmPosition::activate(const trajectory_setpoint_s &last_setpoint)
{
	_sticks_data_required = _param_am_pos_manual_control.get() != 0;

	const bool ret = FlightTask::activate(last_setpoint);
	_yaw_setpoint = NAN;
	_yawspeed_setpoint = 0.f;
	_acceleration_setpoint.setNaN();
	_position_setpoint = _position;
	_velocity_setpoint.zero();
	_setDefaultConstraints();
	updateConstraintsFromEstimator();

	_xy_reset_counter = _sub_vehicle_local_position.get().xy_reset_counter;
	_z_reset_counter = _sub_vehicle_local_position.get().z_reset_counter;

	if (PX4_ISFINITE(_unaided_yaw)) {
		_yaw_error_lpf.reset(matrix::wrap_pi(_yaw - _unaided_yaw));
	}

	_yaw_error_ref = 0.f;
	_yaw_correction = 0.f;
	_yaw_estimate_converging = false;

	return ret;
}

bool FlightTaskManualAmPosition::updateInitialize()
{
	_sticks_data_required = _param_am_pos_manual_control.get() != 0;

	bool ret = FlightTask::updateInitialize();
	_sticks.checkAndUpdateStickInputs();

	if (_sticks_data_required) {
		ret = ret && _sticks.isAvailable();
	}

	return ret
	       && Vector2f(_position).isAllFinite()
	       && Vector2f(_velocity).isAllFinite()
	       && PX4_ISFINITE(_position(2))
	       && PX4_ISFINITE(_velocity(2))
	       && PX4_ISFINITE(_yaw);
}

bool FlightTaskManualAmPosition::update()
{
	bool ret = FlightTask::update();
	updateConstraintsFromEstimator();
	scaleSticks();
	updateSetpoints();
	_constraints.want_takeoff = _checkTakeoff();
	_max_distance_to_ground = INFINITY;
	return ret;
}

void FlightTaskManualAmPosition::_setDefaultConstraints()
{
	_constraints.speed_up = _param_ampc_z_vel_up.get();
	_constraints.speed_down = _param_ampc_z_vel_dn.get();
	_constraints.want_takeoff = false;
}

bool FlightTaskManualAmPosition::_checkTakeoff()
{
	return _sticks.getThrottleZeroCentered() > 0.3f;
}

void FlightTaskManualAmPosition::_ekfResetHandlerHeading(float delta_psi)
{
	if (PX4_ISFINITE(_yaw_setpoint)) {
		_yaw_setpoint = matrix::wrap_pi(_yaw_setpoint + delta_psi);
	}

	_yaw_error_lpf.reset(matrix::wrap_pi(_yaw_error_lpf.getState() + delta_psi));
	_yaw_error_ref = matrix::wrap_pi(_yaw_error_ref + delta_psi);
}

float FlightTaskManualAmPosition::getRollExpo() const
{
	return math::expo_deadzone(_sticks.getRoll(), kExpo, _param_ampc_man_dz.get());
}

float FlightTaskManualAmPosition::getPitchExpo() const
{
	return math::expo_deadzone(_sticks.getPitch(), kExpo, _param_ampc_man_dz.get());
}

float FlightTaskManualAmPosition::getYawExpo() const
{
	return math::expo_deadzone(_sticks.getYaw(), kExpo, _param_ampc_man_dz.get());
}

float FlightTaskManualAmPosition::getThrottleZeroCenteredExpo() const
{
	return math::expo_deadzone(_sticks.getThrottleZeroCentered(), kExpo, _param_ampc_man_dz.get());
}

Vector2f FlightTaskManualAmPosition::getPitchRollExpo() const
{
	return {getPitchExpo(), getRollExpo()};
}

void FlightTaskManualAmPosition::updateConstraintsFromEstimator()
{
	if (PX4_ISFINITE(_sub_vehicle_local_position.get().hagl_min)) {
		_min_distance_to_ground = _sub_vehicle_local_position.get().hagl_min;

	} else {
		_min_distance_to_ground = -INFINITY;
	}

	if (!PX4_ISFINITE(_max_distance_to_ground) && PX4_ISFINITE(_sub_vehicle_local_position.get().hagl_max_z)) {
		_max_distance_to_ground = _sub_vehicle_local_position.get().hagl_max_z;
	}
}

void FlightTaskManualAmPosition::scaleSticks()
{
	const float vel_max_up = fminf(_param_ampc_z_vel_up.get(), _constraints.speed_up);
	const float vel_max_down = fminf(_param_ampc_z_vel_dn.get(), _constraints.speed_down);
	const float vel_max_z = (_sticks.getThrottleZeroCentered() < 0.f) ? vel_max_down : vel_max_up;
	_velocity_setpoint(2) = vel_max_z * -getThrottleZeroCenteredExpo();

	Vector2f stick_xy = getPitchRollExpo();
	Sticks::limitStickUnitLengthXY(stick_xy);

	if (_param_ampc_vel_side.get() >= 0.f && _param_ampc_vel_manual.get() > FLT_EPSILON) {
		stick_xy(1) *= _param_ampc_vel_side.get() / _param_ampc_vel_manual.get();
	}

	if (_param_ampc_vel_back.get() >= 0.f && _param_ampc_vel_manual.get() > FLT_EPSILON && stick_xy(0) < 0.f) {
		stick_xy(0) *= _param_ampc_vel_back.get() / _param_ampc_vel_manual.get();
	}

	float velocity_scale = _param_ampc_vel_manual.get();
	const float max_speed_from_estimator = _sub_vehicle_local_position.get().vxy_max;

	if (PX4_ISFINITE(max_speed_from_estimator)) {
		velocity_scale = math::constrain(velocity_scale, 0.3f, max_speed_from_estimator);
	}

	Vector2f vel_sp_xy = stick_xy * velocity_scale;
	Sticks::rotateIntoHeadingFrameXY(vel_sp_xy, _yaw, _yaw_setpoint);
	_velocity_setpoint.xy() = vel_sp_xy;
}

void FlightTaskManualAmPosition::updateYawSetpoint()
{
	const float yaw_correction_prev = _yaw_correction;
	const bool reset_setpoint = updateYawCorrection(_yaw, _unaided_yaw, _deltatime);

	if (reset_setpoint) {
		_yaw_setpoint = NAN;
	}

	_yawspeed_filter.setParameters(_deltatime, _param_ampc_man_y_tau.get());
	const float yawspeed_scale = math::radians(_param_ampc_man_y_max.get());
	_yawspeed_setpoint = _yawspeed_filter.update(getYawExpo() * yawspeed_scale);
	_yaw_setpoint = updateYawLock(_yaw, _yawspeed_setpoint, _yaw_setpoint, yaw_correction_prev);
}

bool FlightTaskManualAmPosition::updateYawCorrection(float yaw, float unaided_yaw, float dt_s)
{
	if (!PX4_ISFINITE(unaided_yaw)) {
		_yaw_correction = 0.f;
		return false;
	}

	const float yaw_error = matrix::wrap_pi(yaw - unaided_yaw);
	const float yaw_error_hpf = matrix::wrap_pi(yaw_error - _yaw_error_lpf.getState());
	_yaw_error_lpf.update(yaw_error, dt_s);

	const bool was_converging = _yaw_estimate_converging;
	_yaw_estimate_converging = fabsf(yaw_error_hpf) > kYawErrorChangeThreshold;

	bool reset_setpoint = false;

	if (!_yaw_estimate_converging) {
		_yaw_error_ref = yaw_error;

		if (was_converging) {
			reset_setpoint = true;
		}
	}

	_yaw_correction = matrix::wrap_pi(yaw_error - _yaw_error_ref);
	return reset_setpoint;
}

float FlightTaskManualAmPosition::updateYawLock(float yaw, float yawspeed_setpoint, float yaw_setpoint,
		float yaw_correction_prev) const
{
	if (fabsf(yawspeed_setpoint) > FLT_EPSILON) {
		return NAN;
	}

	if (!PX4_ISFINITE(yaw_setpoint)) {
		return yaw;
	}

	return matrix::wrap_pi(yaw_setpoint - yaw_correction_prev + _yaw_correction);
}

void FlightTaskManualAmPosition::updateXYLock()
{
	const float vel_xy_norm = Vector2f(_velocity).length();
	const bool apply_brake = Vector2f(_velocity_setpoint).length() < FLT_EPSILON;
	const bool stopped = (_param_ampc_hold_max_xy.get() < FLT_EPSILON || vel_xy_norm < _param_ampc_hold_max_xy.get());

	if (apply_brake && stopped && !Vector2f(_position_setpoint).isAllFinite()) {
		_position_setpoint.xy() = _position.xy();

	} else if (Vector2f(_position_setpoint).isAllFinite() && apply_brake) {
		if (_sub_vehicle_local_position.get().xy_reset_counter != _xy_reset_counter) {
			_position_setpoint.xy() = _position.xy();
			_xy_reset_counter = _sub_vehicle_local_position.get().xy_reset_counter;
		}

	} else {
		_position_setpoint(0) = NAN;
		_position_setpoint(1) = NAN;
	}
}

void FlightTaskManualAmPosition::updateAltitudeLock()
{
	const bool apply_brake = fabsf(getThrottleZeroCenteredExpo()) <= FLT_EPSILON;
	const bool stopped = (_param_ampc_hold_max_z.get() < FLT_EPSILON || fabsf(_velocity(2)) < _param_ampc_hold_max_z.get());

	if (_param_mpc_alt_mode.get() == 2) {
		const float spd_xy = Vector2f(_velocity).length();
		const bool stick_input = getPitchRollExpo().length() > 0.001f;

		if (_terrain_hold) {
			const bool too_fast = spd_xy > _param_ampc_hold_max_xy.get();

			if (stick_input || too_fast || !PX4_ISFINITE(_dist_to_bottom)) {
				_terrain_hold = false;

				if (PX4_ISFINITE(_dist_to_ground_lock) && PX4_ISFINITE(_dist_to_bottom)) {
					_position_setpoint(2) = _position(2) - (_dist_to_ground_lock - _dist_to_bottom);

				} else {
					_position_setpoint(2) = _position(2);
					_dist_to_ground_lock = NAN;
				}
			}

		} else {
			const bool not_moving = spd_xy < 0.5f * _param_ampc_hold_max_xy.get() && stopped;

			if (!stick_input && not_moving && PX4_ISFINITE(_dist_to_bottom)) {
				_terrain_hold = true;

				if (PX4_ISFINITE(_position_setpoint(2))) {
					_dist_to_ground_lock = _dist_to_bottom - (_position_setpoint(2) - _position(2));
				}
			}
		}
	}

	if ((_param_mpc_alt_mode.get() == 1 || _terrain_hold) && PX4_ISFINITE(_dist_to_bottom)) {
		terrainFollowing(apply_brake, stopped);

	} else {
		if (apply_brake && stopped && !PX4_ISFINITE(_position_setpoint(2))) {
			_position_setpoint(2) = _position(2);

			if (PX4_ISFINITE(_dist_to_bottom) && _dist_to_bottom < _min_distance_to_ground) {
				terrainFollowing(apply_brake, stopped);

			} else {
				_dist_to_ground_lock = NAN;
			}

		} else if (PX4_ISFINITE(_position_setpoint(2)) && apply_brake) {
			if (_sub_vehicle_local_position.get().z_reset_counter != _z_reset_counter) {
				_position_setpoint(2) = _position(2);
				_z_reset_counter = _sub_vehicle_local_position.get().z_reset_counter;
			}

		} else {
			_position_setpoint(2) = NAN;
		}
	}

	respectMaxAltitude();
}

void FlightTaskManualAmPosition::respectMinAltitude()
{
	if (PX4_ISFINITE(_dist_to_bottom) && _dist_to_bottom < _min_distance_to_ground) {
		_position_setpoint(2) = _position(2) - (_min_distance_to_ground - _dist_to_bottom);
	}
}

void FlightTaskManualAmPosition::terrainFollowing(bool apply_brake, bool stopped)
{
	if (apply_brake && stopped && !PX4_ISFINITE(_dist_to_ground_lock)) {
		_position_setpoint(2) = _position(2);
		respectMinAltitude();
		_dist_to_ground_lock = _dist_to_bottom - (_position_setpoint(2) - _position(2));

	} else if (apply_brake && PX4_ISFINITE(_dist_to_ground_lock)) {
		const float delta_distance_to_ground = _dist_to_ground_lock - _dist_to_bottom;
		_position_setpoint(2) = _position(2) - delta_distance_to_ground;

	} else {
		_dist_to_ground_lock = NAN;
		_position_setpoint(2) = NAN;
	}
}

void FlightTaskManualAmPosition::respectMaxAltitude()
{
	if (PX4_ISFINITE(_dist_to_bottom)) {
		const float vel_constrained = _param_mpc_z_p.get() * (_max_distance_to_ground - _dist_to_bottom);

		if (PX4_ISFINITE(_max_distance_to_ground)) {
			_constraints.speed_up = math::constrain(vel_constrained, -_param_ampc_z_vel_dn.get(), _param_ampc_z_vel_up.get());

		} else {
			_constraints.speed_up = _param_ampc_z_vel_up.get();
		}

		if (_dist_to_bottom > _max_distance_to_ground && !(getThrottleZeroCenteredExpo() < FLT_EPSILON)) {
			_velocity_setpoint(2) = math::constrain(-vel_constrained, 0.f, _param_ampc_z_vel_dn.get());
		}

		_constraints.speed_down = _param_ampc_z_vel_dn.get();
	}
}

void FlightTaskManualAmPosition::respectGroundSlowdown()
{
	if (PX4_ISFINITE(_dist_to_ground)) {
		const float limit_down = math::interpolate(_dist_to_ground,
					 _param_mpc_land_alt2.get(), _param_mpc_land_alt1.get(),
					 _param_mpc_land_speed.get(), _constraints.speed_down);
		const float limit_up = math::interpolate(_dist_to_ground,
				       _param_mpc_land_alt2.get(), _param_mpc_land_alt1.get(),
				       _param_mpc_tko_speed.get(), _constraints.speed_up);
		_velocity_setpoint(2) = math::constrain(_velocity_setpoint(2), -limit_up, limit_down);
	}
}

void FlightTaskManualAmPosition::updateSetpoints()
{
	updateYawSetpoint();
	updateAltitudeLock();
	respectGroundSlowdown();
	_acceleration_setpoint.setNaN();
	updateXYLock();

	_weathervane.update();

	if (_weathervane.isActive()) {
		_yaw_setpoint = NAN;

		if (Vector2f(_position_setpoint).isAllFinite()) {
			_yawspeed_setpoint += _weathervane.getWeathervaneYawrate();
		}
	}
}
