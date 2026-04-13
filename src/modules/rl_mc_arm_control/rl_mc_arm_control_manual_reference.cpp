/****************************************************************************
 *
 * Position-style manual reference generator for rl_mc_arm_control.
 *
 ****************************************************************************/

#include "rl_mc_arm_control_manual_reference.hpp"

#include <float.h>
#include <mathlib/math/Functions.hpp>
#include <mathlib/mathlib.h>

using matrix::Dcm2f;
using matrix::Eulerf;
using matrix::Quatf;
using matrix::Vector2f;
using matrix::Vector3f;
using matrix::wrap_pi;

namespace
{
constexpr float kCmdZeroEps = 1e-6f;

Quatf yawOnlyQuat(const float yaw)
{
	return Quatf(Eulerf(0.0f, 0.0f, yaw));
}

void limitStickUnitLengthXY(Vector2f &stick_xy)
{
	const float stick_xy_norm = stick_xy.length();

	if (stick_xy_norm > 1.0f) {
		stick_xy /= stick_xy_norm;
	}
}

void rotateIntoHeadingFrameXY(Vector2f &stick_xy, const float yaw, const float yaw_sp)
{
	const float yaw_rotate = PX4_ISFINITE(yaw_sp) ? yaw_sp : yaw;
	stick_xy = Dcm2f(yaw_rotate) * stick_xy;
}

} // namespace

void RlMcArmControlManualReference::reset()
{
	_initialized = false;
	_hold_pos_xy = Vector2f(NAN, NAN);
	_yaw_sp = NAN;
	_yawspeed_filter.reset(0.0f);
}

void RlMcArmControlManualReference::initializeIfNeeded(const Input &input)
{
	if (_initialized) {
		return;
	}

	_xy_smoothing.reset(Vector2f(), input.root_lin_vel_w.xy(), input.root_pos_w.xy());
	_xy_smoothing.setVelSpFeedback(input.root_lin_vel_w.xy());
	_xy_smoothing.setCurrentPositionEstimate(input.root_pos_w.xy());
	_z_smoothing.reset(0.0f, input.velocity_z_ned, input.position_z_ned);
	_z_smoothing.setVelSpFeedback(input.velocity_z_ned);
	_z_smoothing.setCurrentPositionEstimate(input.position_z_ned);
	_yaw_sp = input.heading_w;
	_initialized = true;
}

RlMcArmControlManualReference::Output RlMcArmControlManualReference::update(const Params &params, const Input &input)
{
	initializeIfNeeded(input);

	Output output{};
	output.desired_pos_w = input.root_pos_w;
	output.desired_quat_w = yawOnlyQuat(input.heading_w);

	const float dt = math::constrain(input.dt_s, 0.0002f, 0.02f);
	const float deadzone = math::constrain(params.man_deadzone, 0.0f, 0.4f);
	const float vel_forward = math::max(params.vel_manual, kCmdZeroEps);
	const float vel_side = (params.vel_side >= 0.0f && params.vel_side <= vel_forward) ? params.vel_side : vel_forward;
	const float vel_back = (params.vel_back >= 0.0f && params.vel_back <= vel_forward) ? params.vel_back : vel_forward;

	float roll = 0.0f;
	float pitch = 0.0f;
	float yaw = 0.0f;
	float throttle = 0.0f;

	if (input.sticks_available
	    && PX4_ISFINITE(input.roll)
	    && PX4_ISFINITE(input.pitch)
	    && PX4_ISFINITE(input.yaw)
	    && PX4_ISFINITE(input.throttle)) {
		roll = math::expo_deadzone(input.roll, 0.6f, deadzone);
		pitch = math::expo_deadzone(input.pitch, 0.6f, deadzone);
		yaw = math::expo_deadzone(input.yaw, 0.6f, deadzone);
		throttle = math::expo_deadzone(input.throttle, 0.6f, deadzone);
	}

	output.processed_roll = roll;
	output.processed_pitch = pitch;
	output.processed_yaw = yaw;
	output.processed_throttle = throttle;

	Vector2f stick_xy{pitch, roll};
	limitStickUnitLengthXY(stick_xy);

	if (params.vel_side >= 0.0f) {
		stick_xy(1) *= vel_side / vel_forward;
	}

	if (params.vel_back >= 0.0f && stick_xy(0) < 0.0f) {
		stick_xy(0) *= vel_back / vel_forward;
	}

	Vector2f target_vel_xy = stick_xy * vel_forward;
	rotateIntoHeadingFrameXY(target_vel_xy, input.heading_w, _yaw_sp);

	_xy_smoothing.setMaxJerk(math::max(params.jerk_max, 0.0f));
	_xy_smoothing.setMaxAccel(math::max(params.acc_hor, 0.0f));
	_xy_smoothing.setMaxVel(vel_forward);
	_xy_smoothing.setVelSpFeedback(input.root_lin_vel_w.xy());
	_xy_smoothing.setCurrentPositionEstimate(input.root_pos_w.xy());
	_xy_smoothing.update(dt, target_vel_xy);

	const Vector2f desired_vel_xy_w = _xy_smoothing.getCurrentVelocity();
	const float hold_xy_threshold = math::max(params.hold_max_xy, 0.0f);

	if (target_vel_xy.length() > FLT_EPSILON) {
		_hold_pos_xy = Vector2f(NAN, NAN);

	} else {
		const Vector2f root_vel_xy_w{input.root_lin_vel_w(0), input.root_lin_vel_w(1)};

		if (!_hold_pos_xy.isAllFinite()
		    && desired_vel_xy_w.length() <= hold_xy_threshold
		    && root_vel_xy_w.length() <= hold_xy_threshold) {
			_hold_pos_xy = input.root_pos_w.xy();
		}
	}

	const float target_vel_z_ned = (throttle < 0.0f) ? (-throttle * params.z_vel_up) : (-throttle * params.z_vel_down);
	_z_smoothing.setMaxJerk(math::max(params.jerk_max, 0.0f));
	_z_smoothing.setMaxAccelUp(math::max(params.acc_hor, 0.0f));
	_z_smoothing.setMaxAccelDown(math::max(params.acc_hor, 0.0f));
	_z_smoothing.setMaxVelUp(math::max(params.z_vel_up, 0.0f));
	_z_smoothing.setMaxVelDown(math::max(params.z_vel_down, 0.0f));
	_z_smoothing.setVelSpFeedback(input.velocity_z_ned);
	_z_smoothing.setCurrentPositionEstimate(input.position_z_ned);
	_z_smoothing.update(dt, target_vel_z_ned);

	_yawspeed_filter.setParameters(dt, math::max(params.man_y_tau, 0.0f));
	const float filtered_yaw_rate = _yawspeed_filter.update(yaw * math::max(params.man_y_max_rad, 0.0f));

	if (fabsf(filtered_yaw_rate) > kCmdZeroEps) {
		_yaw_sp = NAN;
		output.desired_ang_vel_b(2) = filtered_yaw_rate;

	} else {
		if (!PX4_ISFINITE(_yaw_sp)) {
			_yaw_sp = input.heading_w;
		}
	}

	const Vector3f desired_vel_w{desired_vel_xy_w(0), desired_vel_xy_w(1), -_z_smoothing.getCurrentVelocity()};
	output.desired_lin_vel_b = input.root_quat_w.inversed().rotateVector(desired_vel_w);

	if (_hold_pos_xy.isAllFinite()) {
		output.desired_pos_w(0) = _hold_pos_xy(0);
		output.desired_pos_w(1) = _hold_pos_xy(1);
	}

	const float desired_pos_z_ned = _z_smoothing.getCurrentPosition();
	const bool hold_z_active = fabsf(_z_smoothing.getCurrentVelocity()) <= math::max(params.hold_max_z, 0.0f)
				   && fabsf(throttle) <= FLT_EPSILON
				   && PX4_ISFINITE(desired_pos_z_ned);
	if (hold_z_active) {
		output.desired_pos_w(2) = -desired_pos_z_ned;
	}

	output.desired_quat_w = yawOnlyQuat(PX4_ISFINITE(_yaw_sp) ? _yaw_sp : input.heading_w);
	output.lin_cmd_active = {{
		fabsf(output.desired_lin_vel_b(0)) > kCmdZeroEps,
		fabsf(output.desired_lin_vel_b(1)) > kCmdZeroEps,
		fabsf(output.desired_lin_vel_b(2)) > kCmdZeroEps,
	}};
	output.ang_cmd_active = {{false, false, fabsf(output.desired_ang_vel_b(2)) > kCmdZeroEps}};
	output.hold_xy_active = _hold_pos_xy.isAllFinite();
	output.hold_z_active = hold_z_active;
	output.hold_yaw_active = PX4_ISFINITE(_yaw_sp) && fabsf(output.desired_ang_vel_b(2)) <= kCmdZeroEps;

	return output;
}
