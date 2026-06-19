/****************************************************************************
 *
 * AM Position direct actuator controller.
 *
 ****************************************************************************/

#include "am_pos_control.hpp"

#include <cmath>
#include <cstring>

namespace
{
const matrix::Vector3f kGravityEnu{0.0f, 0.0f, -1.0f};
constexpr float kHalfSqrt2 = 0.7071067811865476f;

float wrapToPi(float angle)
{
	while (angle > M_PI_F) {
		angle -= 2.f * M_PI_F;
	}

	while (angle < -M_PI_F) {
		angle += 2.f * M_PI_F;
	}

	return angle;
}

matrix::Quatf yawOnlyQuat(float yaw)
{
	return matrix::Quatf(matrix::Eulerf(0.f, 0.f, yaw));
}

matrix::Vector3f positionNedToEnu(const matrix::Vector3f &ned)
{
	return {ned(1), ned(0), -ned(2)};
}

matrix::Vector3f positionEnuToNed(const matrix::Vector3f &enu)
{
	return {enu(1), enu(0), -enu(2)};
}

matrix::Vector3f frdToFlu(const matrix::Vector3f &frd)
{
	return {frd(0), -frd(1), -frd(2)};
}

matrix::Quatf attitudeNedToEnu(const matrix::Quatf &q_ned)
{
	const matrix::Quatf q_ned_to_enu{0.0f, kHalfSqrt2, kHalfSqrt2, 0.0f};
	const matrix::Quatf q_flu_to_frd{0.0f, 1.0f, 0.0f, 0.0f};
	return q_ned_to_enu * q_ned * q_flu_to_frd;
}

float yawNedToEnu(float yaw_ned_rad)
{
	return wrapToPi(M_PI_F / 2.0f - yaw_ned_rad);
}
}

AmPosControl::AmPosControl() :
	ModuleParams(nullptr),
	WorkItem(MODULE_NAME, px4::wq_configurations::nav_and_controllers),
	_loop_perf(perf_alloc(PC_ELAPSED, MODULE_NAME": cycle"))
{
	resetState();
}

AmPosControl::~AmPosControl()
{
	perf_free(_loop_perf);
}

bool AmPosControl::init()
{
	if (!_angular_velocity_sub.registerCallback()) {
		PX4_ERR("callback registration failed");
		return false;
	}

	_takeoff_status_pub.advertise();
	_takeoff.setSpoolupTime(_param_com_spoolup_time.get());
	_takeoff.setTakeoffRampTime(_param_mpc_tko_ramp_t.get());

	return _adapter.init();
}

void AmPosControl::resetCommandReference()
{
	_current_cmd_ref = {};
	_manual_yaw_release_start = 0;
}

void AmPosControl::resetState()
{
	_trajectory_setpoint = {};
	_position = {};
	_attitude = {};
	_angular_velocity = {};
	_arm_joint_state = {};
	_vehicle_constraints = {0, NAN, NAN, false, {}};
	_vehicle_control_mode = {};
	_vehicle_land_detected = {
		.timestamp = 0,
		.freefall = false,
		.ground_contact = true,
		.maybe_landed = true,
		.landed = true,
	};
	resetActionHistory();
	_root_pos_w.zero();
	_root_quat_w = matrix::Quatf();
	_root_lin_vel_w.zero();
	_root_lin_vel_b.zero();
	_root_ang_vel_b.zero();
	_heading_w = 0.0f;

	for (float &arm_position : _normalized_arm_position) {
		arm_position = 0.0f;
	}

	_arm_joint_state_position_wrap_mask = 0;
	_takeoff_ramped_speed_up = 0.0f;
	_takeoff_target_speed_up = 0.0f;
	_manual_yaw_release_start = 0;
	_am_offboard_using_external_setpoint = false;
	_policy_sequence = 0;
	resetCommandReference();
	_adapter.reset();
}

void AmPosControl::publishTakeoffStatus()
{
	const uint8_t takeoff_state = static_cast<uint8_t>(_takeoff.getTakeoffState());

	if (takeoff_state != _takeoff_status_pub.get().takeoff_state) {
		_takeoff_status_pub.get().takeoff_state = takeoff_state;
		_takeoff_status_pub.get().tilt_limit = NAN;
		_takeoff_status_pub.get().timestamp = hrt_absolute_time();
		_takeoff_status_pub.update();
	}
}

bool AmPosControl::updateTakeoffGate(ActiveMode mode, bool was_using_am_mode, float dt_s)
{
	_vehicle_control_mode_sub.update(&_vehicle_control_mode);
	_vehicle_land_detected_sub.update(&_vehicle_land_detected);
	_vehicle_constraints_sub.update(&_vehicle_constraints);

	bool want_takeoff = _vehicle_constraints.want_takeoff;
	float speed_up = PX4_ISFINITE(_vehicle_constraints.speed_up) ? _vehicle_constraints.speed_up :
			 _param_ampc_z_vel_up.get();

	if (mode == ActiveMode::Offboard) {
		want_takeoff = offboardSetpointWantsTakeoff(_trajectory_setpoint, _position, _position.timestamp_sample);
		speed_up = _param_ampc_z_vel_up.get();
	}

	_takeoff_target_speed_up = speed_up;

	const bool skip_takeoff = shouldSkipTakeoffRampOnAmModeEntry(_use_am_mode, was_using_am_mode,
				  _vehicle_land_detected.landed);
	_takeoff.updateTakeoffState(_vehicle_control_mode.flag_armed, _vehicle_land_detected.landed,
				    want_takeoff, speed_up, skip_takeoff, _position.timestamp_sample);

	_takeoff_ramped_speed_up = _takeoff.updateRamp(dt_s, speed_up);

	publishTakeoffStatus();

	return takeoffStateRequiresOutputGate(static_cast<uint8_t>(_takeoff.getTakeoffState()),
					      _vehicle_land_detected.ground_contact);
}

bool AmPosControl::anyAxisActive(const bool axes[3]) const
{
	return axes[0] || axes[1] || axes[2];
}

bool AmPosControl::updateVehicleState(float &dt_s)
{
	if (!_angular_velocity_sub.update(&_angular_velocity)) {
		return false;
	}

	dt_s = math::constrain((_angular_velocity.timestamp_sample - _last_run) * 1e-6f, 0.0002f, 0.02f);
	_last_run = _angular_velocity.timestamp_sample;

	_attitude_sub.update(&_attitude);
	_position_sub.update(&_position);
	_arm_joint_state_sub.update(&_arm_joint_state);

	if (!vehicleStateValid()) {
		return false;
	}

	updateConvertedState();
	return true;
}

bool AmPosControl::attitudeValid() const
{
	const hrt_abstime now = hrt_absolute_time();

	if ((_attitude.timestamp == 0) || (now > _attitude.timestamp + kStateTimeout)) {
		return false;
	}

	const matrix::Quatf q{_attitude.q};
	const float norm = q.norm();
	return PX4_ISFINITE(q(0)) && PX4_ISFINITE(q(1)) && PX4_ISFINITE(q(2)) && PX4_ISFINITE(q(3))
	       && PX4_ISFINITE(norm) && (fabsf(1.f - norm) <= 1e-3f);
}

bool AmPosControl::angularVelocityValid() const
{
	const hrt_abstime now = hrt_absolute_time();

	if ((_angular_velocity.timestamp == 0) || (now > _angular_velocity.timestamp + kStateTimeout)) {
		return false;
	}

	return PX4_ISFINITE(_angular_velocity.xyz[0]) && PX4_ISFINITE(_angular_velocity.xyz[1])
	       && PX4_ISFINITE(_angular_velocity.xyz[2]);
}

bool AmPosControl::vehicleStateValid() const
{
	return vehicleStateValidStrict(_position, attitudeValid(), angularVelocityValid(), hrt_absolute_time());
}

bool AmPosControl::armStateValid() const
{
	const hrt_abstime now = hrt_absolute_time();

	if ((_arm_joint_state.timestamp == 0) || (now > _arm_joint_state.timestamp + kArmStateTimeout)) {
		return false;
	}

	for (int i = 0; i < kArmJointDim; ++i) {
		if (!PX4_ISFINITE(_arm_joint_state.arm_position[i])) {
			return false;
		}
	}

	return true;
}

void AmPosControl::updateNormalizedArmJointState()
{
	_arm_joint_state_position_wrap_mask = 0;

	for (int i = 0; i < kArmJointDim; ++i) {
		_normalized_arm_position[i] = wrapArmJointPositionForPolicy(_arm_joint_state.arm_position[i],
					     _arm_joint_state_position_wrap_mask, i);
	}
}

bool AmPosControl::trajectorySetpointValid() const
{
	return trajectorySetpointFresh() && trajectorySetpointHasLinearInput();
}

bool AmPosControl::trajectorySetpointFresh() const
{
	const hrt_abstime now = hrt_absolute_time();

	if ((_trajectory_setpoint.timestamp == 0) || (now > _trajectory_setpoint.timestamp + kTrajectorySetpointTimeout)) {
		return false;
	}

	return true;
}

bool AmPosControl::trajectorySetpointHasLinearInput() const
{
	return PX4_ISFINITE(_trajectory_setpoint.position[0]) || PX4_ISFINITE(_trajectory_setpoint.position[1])
	       || PX4_ISFINITE(_trajectory_setpoint.position[2]) || PX4_ISFINITE(_trajectory_setpoint.velocity[0])
	       || PX4_ISFINITE(_trajectory_setpoint.velocity[1]) || PX4_ISFINITE(_trajectory_setpoint.velocity[2]);
}

bool AmPosControl::offboardTrajectorySetpointValid() const
{
	if (!trajectorySetpointFresh()) {
		return false;
	}

	if (_offboard_control_mode.velocity) {
		return PX4_ISFINITE(_trajectory_setpoint.velocity[0]) || PX4_ISFINITE(_trajectory_setpoint.velocity[1])
		       || PX4_ISFINITE(_trajectory_setpoint.velocity[2]);
	}

	return false;
}

bool AmPosControl::offboardControlModeFresh() const
{
	const hrt_abstime now = hrt_absolute_time();
	const hrt_abstime timeout = static_cast<hrt_abstime>(_param_com_of_loss_t.get() * 1_s);
	return (_offboard_control_mode.timestamp != 0) && (now <= _offboard_control_mode.timestamp + timeout);
}

bool AmPosControl::offboardControlModeSupported() const
{
	return offboardControlModeSupported(_offboard_control_mode);
}

bool AmPosControl::offboardControlModeValid() const
{
	return offboardControlModeFresh() && offboardControlModeSupported();
}

bool AmPosControl::manualControlAvailable()
{
	_sticks.checkAndUpdateStickInputs();
	return _sticks.isAvailable();
}

AmPosControl::ActiveMode AmPosControl::activeMode()
{
	vehicle_status_s vehicle_status{};

	if (_vehicle_status_sub.copy(&vehicle_status)) {
		if (vehicle_status.nav_state == vehicle_status_s::NAVIGATION_STATE_AM_POSITION) {
			return ActiveMode::Manual;
		}

		if (vehicle_status.nav_state == vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD) {
			return ActiveMode::Offboard;
		}
	}

	return ActiveMode::None;
}

void AmPosControl::publishStatus()
{
	_offboard_control_mode_sub.update(&_offboard_control_mode);
	_arm_joint_state_sub.update(&_arm_joint_state);
	_trajectory_setpoint_sub.update(&_trajectory_setpoint);

	am_pos_control_status_s status{};
	status.timestamp = hrt_absolute_time();
	status.module_running = true;
	status.manual_control_available = manualControlAvailable();
	status.vehicle_state_valid = vehicleStateValid();
	status.attitude_valid = attitudeValid();
	status.angular_velocity_valid = angularVelocityValid();
	status.arm_state_valid = armStateValid();
	status.trajectory_setpoint_valid = trajectorySetpointValid();
	status.offboard_control_mode_fresh = offboardControlModeFresh();
	status.offboard_control_mode_supported = offboardControlModeSupported();
	status.offboard_control_mode_valid = offboardControlModeValid();
	status.am_position_available = status.manual_control_available && status.arm_state_valid;
	status.am_offboard_available = status.arm_state_valid;
	_status_pub.publish(status);
}

void AmPosControl::updateConvertedState()
{
	const matrix::Vector3f pos_ned(_position.x, _position.y, _position.z);
	const matrix::Vector3f vel_ned(_position.vx, _position.vy, _position.vz);
	const matrix::Quatf q_ned(_attitude.q);
	const matrix::Vector3f ang_vel_frd(_angular_velocity.xyz[0], _angular_velocity.xyz[1], _angular_velocity.xyz[2]);

	_root_pos_w = positionNedToEnu(pos_ned);
	_root_lin_vel_w = positionNedToEnu(vel_ned);
	_root_quat_w = attitudeNedToEnu(q_ned);
	_heading_w = yawNedToEnu(_position.heading);
	_root_lin_vel_b = _root_quat_w.inversed().rotateVector(_root_lin_vel_w);
	_root_ang_vel_b = frdToFlu(ang_vel_frd);
}

void AmPosControl::updateTargets()
{
	updateTargets(false);
}

void AmPosControl::updateTargets(bool respect_trajectory_yaw)
{
	_trajectory_setpoint_sub.update(&_trajectory_setpoint);
	const ActiveMode mode = activeMode();

	matrix::Vector3f desired_vel_ned{};

	for (int i = 0; i < 3; ++i) {
		desired_vel_ned(i) = PX4_ISFINITE(_trajectory_setpoint.velocity[i]) ? _trajectory_setpoint.velocity[i] : 0.0f;
	}

	if (_takeoff.getTakeoffState() == TakeoffState::rampup) {
		desired_vel_ned(2) = constrainUpwardVelocityNed(desired_vel_ned(2), _takeoff_ramped_speed_up);

		if (PX4_ISFINITE(_trajectory_setpoint.position[2]) && PX4_ISFINITE(_position.z)
		    && (_trajectory_setpoint.position[2] < _position.z)) {
			desired_vel_ned(2) = constrainUpwardVelocityNed(-_takeoff_ramped_speed_up, _takeoff_ramped_speed_up);
		}
	}

	const matrix::Vector3f desired_vel_w = positionNedToEnu(desired_vel_ned);
	matrix::Vector3f desired_ang_vel_w{};
	desired_ang_vel_w.zero();

	if (!_current_cmd_ref.initialized) {
		_current_cmd_ref.desired_pos_w = _root_pos_w;
		_current_cmd_ref.desired_quat_w = yawOnlyQuat(_heading_w);
		_current_cmd_ref.initialized = true;
	}

	const bool yaw_active = yawRateActiveForMode(_trajectory_setpoint.yawspeed, _current_cmd_ref.yaw_cmd_active,
				mode, _manual_yaw_release_start, hrt_absolute_time());

	if (yaw_active && PX4_ISFINITE(_trajectory_setpoint.yawspeed)) {
		desired_ang_vel_w(2) = -_trajectory_setpoint.yawspeed;
	}

	const float previous_desired_yaw_w = matrix::Eulerf(_current_cmd_ref.desired_quat_w).psi();
	const float trajectory_yaw_w = PX4_ISFINITE(_trajectory_setpoint.yaw) ? yawNedToEnu(_trajectory_setpoint.yaw) : NAN;
	const float desired_yaw_w = desiredYawForCommandReference(previous_desired_yaw_w, _heading_w, trajectory_yaw_w,
				    yaw_active, _current_cmd_ref.yaw_cmd_active, respect_trajectory_yaw);

	const matrix::Quatf desired_quat_w = yawOnlyQuat(desired_yaw_w);
	const matrix::Vector3f desired_vel_h = desired_quat_w.inversed().rotateVector(desired_vel_w);
	bool lin_active_h[3] {};
	activeAxesFromCommand(desired_vel_h, lin_active_h);

	if (_takeoff.getTakeoffState() == TakeoffState::rampup) {
		lin_active_h[2] = true;
	}

	matrix::Vector3f desired_pos_ned = positionEnuToNed(_current_cmd_ref.desired_pos_w);

	for (int i = 0; i < 3; ++i) {
		if (PX4_ISFINITE(_trajectory_setpoint.position[i])) {
			desired_pos_ned(i) = _trajectory_setpoint.position[i];
		}
	}

	matrix::Vector3f desired_pos_w = positionNedToEnu(desired_pos_ned);
	matrix::Vector3f pos_error_w = desired_pos_w - _root_pos_w;
	matrix::Vector3f pos_error_h = desired_quat_w.inversed().rotateVector(pos_error_w);

	for (int i = 0; i < 3; ++i) {
		if (lin_active_h[i]) {
			pos_error_h(i) = 0.0f;
		}
	}

	desired_pos_w = _root_pos_w + desired_quat_w.rotateVector(pos_error_h);

	_current_cmd_ref.desired_lin_vel_w = desired_vel_w;
	_current_cmd_ref.desired_ang_vel_w = desired_ang_vel_w;
	_current_cmd_ref.desired_pos_w = desired_pos_w;
	_current_cmd_ref.desired_quat_w = desired_quat_w;

	for (int i = 0; i < 3; ++i) {
		_current_cmd_ref.lin_cmd_active_h[i] = lin_active_h[i];
	}

	_current_cmd_ref.yaw_cmd_active = yaw_active;
	_current_cmd_ref.has_lin_vel_cmd = anyAxisActive(lin_active_h);
	_current_cmd_ref.has_ang_vel_cmd = yaw_active;
}

void AmPosControl::buildObservation(AmPolicyAdapter::Observation &observation)
{
	for (int i = 0; i < AmPolicyAdapter::ObservationDim; ++i) {
		observation[i] = 0.0f;
	}

	const matrix::Vector3f &root_pos_w = _root_pos_w;
	const matrix::Quatf &root_quat_w = _root_quat_w;
	const matrix::Vector3f &lin_vel_b = _root_lin_vel_b;
	const matrix::Vector3f &ang_vel_b = _root_ang_vel_b;
	const matrix::Vector3f gated_pos_err_b = gatePositionErrorForPolicy(
				_current_cmd_ref.desired_pos_w - root_pos_w, root_quat_w,
				_current_cmd_ref.desired_quat_w, _current_cmd_ref.lin_cmd_active_h);

	const float desired_yaw_w = matrix::Eulerf(_current_cmd_ref.desired_quat_w).psi();
	const matrix::Quatf desired_att_w(matrix::Eulerf(0.f, 0.f, desired_yaw_w));
	const matrix::Dcmf att_err_dcm(attitudeErrorQuatForPolicy(root_quat_w, desired_att_w,
						_current_cmd_ref.yaw_cmd_active));
	const matrix::Vector3f projected_gravity_b = root_quat_w.inversed().rotateVector(kGravityEnu);
	const matrix::Vector3f lin_vel_err_b = linearVelocityErrorForPolicy(
			_current_cmd_ref.desired_lin_vel_w, root_quat_w.rotateVector(lin_vel_b),
			root_quat_w, _current_cmd_ref.desired_quat_w, _current_cmd_ref.lin_cmd_active_h);
	const matrix::Vector3f ang_vel_err_b = angularVelocityErrorForPolicy(
			_current_cmd_ref.desired_ang_vel_w, root_quat_w.rotateVector(ang_vel_b),
			root_quat_w, _current_cmd_ref.yaw_cmd_active);

	int idx = 0;
	observation[idx++] = gated_pos_err_b(0);
	observation[idx++] = gated_pos_err_b(1);
	observation[idx++] = gated_pos_err_b(2);

	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			observation[idx++] = att_err_dcm(r, c);
		}
	}

	observation[idx++] = projected_gravity_b(0);
	observation[idx++] = projected_gravity_b(1);
	observation[idx++] = projected_gravity_b(2);
	observation[idx++] = lin_vel_err_b(0);
	observation[idx++] = lin_vel_err_b(1);
	observation[idx++] = lin_vel_err_b(2);
	observation[idx++] = ang_vel_err_b(0);
	observation[idx++] = ang_vel_err_b(1);
	observation[idx++] = ang_vel_err_b(2);

	for (int i = 0; i < kArmJointDim; ++i) {
		observation[idx++] = _normalized_arm_position[i];
	}

	for (int i = 0; i < kActionDim; ++i) {
		observation[idx++] = _prev_action[i];
	}
}

void AmPosControl::updateActionHistory(const AmPolicyAdapter::Action &action)
{
	for (int i = 0; i < kActionDim; ++i) {
		_prev_action[i] = action[i];
	}
}

void AmPosControl::resetActionHistory()
{
	for (int i = 0; i < kActionDim; ++i) {
		_prev_action[i] = 0.0f;
	}
}

void AmPosControl::maybeLogPolicyDiagnostics(const AmPolicyAdapter::Observation &observation,
		const AmPolicyAdapter::Action &action, const AmPolicyAdapter::Action &executed_action)
{
	if (_startup_diag_samples_remaining <= 0) {
		return;
	}

	const ActiveMode mode = activeMode();
	const char *mode_label = mode == ActiveMode::Offboard ? "AM Offboard" : "AM Position";

	float motor_control[kActionDim] {};
	float motor_sum = 0.0f;
	float motor_min = INFINITY;
	float motor_max = -INFINITY;

	for (int i = 0; i < kActionDim; ++i) {
		motor_control[i] = clampNormalizedMotorControl(executed_action[i]);
		motor_sum += motor_control[i];
		motor_min = math::min(motor_min, motor_control[i]);
		motor_max = math::max(motor_max, motor_control[i]);
	}

	const float motor_mean = motor_sum / static_cast<float>(kActionDim);
	const float motor_spread = motor_max - motor_min;

	PX4_INFO(
		"%s obs pos_err=(%.3f, %.3f, %.3f) grav_z=%.3f prev_action=(%.3f, %.3f, %.3f, %.3f)",
		mode_label,
		(double)observation[0], (double)observation[1], (double)observation[2], (double)observation[14],
		(double)observation[26], (double)observation[27], (double)observation[28], (double)observation[29]);
	PX4_INFO(
		"%s act raw=(%.3f, %.3f, %.3f, %.3f) motor=(%.3f, %.3f, %.3f, %.3f) mean=%.3f spread=%.3f",
		mode_label,
		(double)action[0], (double)action[1], (double)action[2], (double)action[3],
		(double)motor_control[0], (double)motor_control[1], (double)motor_control[2], (double)motor_control[3],
		(double)motor_mean, (double)motor_spread);

	if (motor_mean < kDiagLowMeanCommand
	    || (motor_min < kDiagLowMinCommand && motor_spread > kDiagWideSpread)) {
		PX4_WARN(
			"%s motor command distribution looks weak/imbalanced: mean=%.3f min=%.3f max=%.3f",
			mode_label,
			(double)motor_mean, (double)motor_min, (double)motor_max);
	}

	--_startup_diag_samples_remaining;
}

void AmPosControl::publishPolicyObservation(const AmPolicyAdapter::Observation &observation,
		const AmPolicyAdapter::Action &action, const actuator_motors_s &actuator_motors, uint32_t degraded_flags,
		const PolicyObservationTiming &timing)
{
	am_policy_observation_s policy_observation{};
	fillPolicyObservation(policy_observation, observation, action, actuator_motors, degraded_flags, timing);
	_policy_observation_pub.publish(policy_observation);
}

void AmPosControl::applyAction(const AmPolicyAdapter::Observation &observation, const AmPolicyAdapter::Action &action,
				       AmPolicyAdapter::Action &executed_action, ActiveMode mode, bool publish_outputs,
				       uint32_t degraded_flags, const PolicyObservationTiming &timing)
{
	actuator_motors_s actuator_motors{};
	const hrt_abstime now = hrt_absolute_time();
	fillMotorSetpointFromAction(actuator_motors, executed_action, action, now, now);
	publishPolicyObservation(observation, action, actuator_motors, degraded_flags, timing);

	if (publish_outputs) {
		_actuator_motors_pub.publish(actuator_motors);

		vehicle_thrust_setpoint_s thrust_setpoint{};
		fillThrustSetpointFromMotors(thrust_setpoint, actuator_motors.timestamp, actuator_motors.timestamp_sample,
					     actuator_motors);
		_vehicle_thrust_setpoint_pub.publish(thrust_setpoint);
	}
}

void AmPosControl::publishStopSetpoint()
{
	actuator_motors_s actuator_motors{};
	actuator_motors.timestamp = hrt_absolute_time();
	actuator_motors.timestamp_sample = actuator_motors.timestamp;

	for (int i = 0; i < kMotorControlDim; ++i) {
		actuator_motors.control[i] = NAN;
	}

	actuator_motors.reversible_flags = 0;
	_actuator_motors_pub.publish(actuator_motors);

	vehicle_thrust_setpoint_s thrust_setpoint{};
	fillThrustSetpointFromMotors(thrust_setpoint, actuator_motors.timestamp, actuator_motors.timestamp_sample,
				     actuator_motors);
	_vehicle_thrust_setpoint_pub.publish(thrust_setpoint);
}

void AmPosControl::publishIdleSetpoint()
{
	actuator_motors_s actuator_motors{};
	const hrt_abstime now = hrt_absolute_time();
	fillIdleMotorSetpoint(actuator_motors, now, now);
	_actuator_motors_pub.publish(actuator_motors);

	vehicle_thrust_setpoint_s thrust_setpoint{};
	fillThrustSetpointFromMotors(thrust_setpoint, actuator_motors.timestamp, actuator_motors.timestamp_sample,
				     actuator_motors);
	_vehicle_thrust_setpoint_pub.publish(thrust_setpoint);
}

void AmPosControl::Run()
{
	if (should_exit()) {
		publishStopSetpoint();
		_angular_velocity_sub.unregisterCallback();
		exit_and_cleanup();
		return;
	}

	perf_begin(_loop_perf);

	vehicle_status_s vehicle_status{};
	const bool was_using_am_mode = _use_am_mode;

	if (_vehicle_status_sub.updated()) {
		_vehicle_status_sub.copy(&vehicle_status);
		_use_am_mode = vehicle_status.nav_state == vehicle_status_s::NAVIGATION_STATE_AM_POSITION
			       || vehicle_status.nav_state == vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD;
	}

	const ActiveMode mode = activeMode();

	if (_parameter_update_sub.updated()) {
		parameter_update_s param_update{};
		_parameter_update_sub.copy(&param_update);
		updateParams();
		_takeoff.setSpoolupTime(_param_com_spoolup_time.get());
		_takeoff.setTakeoffRampTime(_param_mpc_tko_ramp_t.get());
	}

	float dt_s = 0.0f;
	const bool vehicle_state_updated = updateVehicleState(dt_s);
	publishStatus();

	if (_use_am_mode && !was_using_am_mode) {
		resetCommandReference();
		resetActionHistory();
		_takeoff_ramped_speed_up = 0.0f;
		_takeoff_target_speed_up = 0.0f;
		_am_offboard_using_external_setpoint = false;
		_adapter.reset();
		_startup_diag_samples_remaining = kStartupDiagSamples;
	}

	if (!_use_am_mode) {
		if (was_using_am_mode) {
			publishStopSetpoint();
			resetCommandReference();
			resetActionHistory();
			_takeoff_ramped_speed_up = 0.0f;
			_takeoff_target_speed_up = 0.0f;
			_am_offboard_using_external_setpoint = false;
			_adapter.reset();
		}

		perf_end(_loop_perf);
		return;
	}

	uint32_t am_policy_degraded_flags = am_policy_observation_s::DEGRADED_NONE;

	if (!vehicle_state_updated) {
		_adapter.reset();
		resetActionHistory();
		resetCommandReference();
		publishStopSetpoint();
		perf_end(_loop_perf);
		return;
	}

	if (!armStateValid()) {
		_adapter.reset();
		resetActionHistory();
		resetCommandReference();
		publishStopSetpoint();
		perf_end(_loop_perf);
		return;
	}

	updateNormalizedArmJointState();

	_trajectory_setpoint_sub.update(&_trajectory_setpoint);
	_offboard_control_mode_sub.update(&_offboard_control_mode);

	bool am_offboard_using_external_setpoint = false;

	if (mode == ActiveMode::Offboard) {
		am_offboard_using_external_setpoint =
			amOffboardExternalSetpointUsable(_offboard_control_mode, _trajectory_setpoint, offboardControlModeValid(),
					trajectorySetpointFresh());

		if (!am_offboard_using_external_setpoint) {
			fillAmOffboardHoldSetpoint(_trajectory_setpoint, hrt_absolute_time(), am_policy_degraded_flags);

			if (_am_offboard_using_external_setpoint) {
				resetCommandReference();
			}
		}
	}

	const bool trajectory_setpoint_valid = mode == ActiveMode::Offboard ? true : trajectorySetpointValid();

	if (!trajectory_setpoint_valid) {
		if (hrt_elapsed_time(&_last_setpoint_diag) > 1_s) {
			_last_setpoint_diag = hrt_absolute_time();
			PX4_WARN("AM Position waiting for fresh trajectory_setpoint");
		}

		_adapter.reset();
		resetActionHistory();
		resetCommandReference();
		publishStopSetpoint();
		perf_end(_loop_perf);
		return;
	}

	if (updateTakeoffGate(mode, was_using_am_mode, dt_s)) {
		_adapter.reset();
		resetCommandReference();
		resetActionHistory();
		publishIdleSetpoint();
		perf_end(_loop_perf);
		return;
	}

	updateTargets(mode != ActiveMode::Manual);

	const bool commit_policy_state = takeoffStateAllowsPolicyStateCommit(static_cast<uint8_t>(_takeoff.getTakeoffState()));

	if (!commit_policy_state) {
		_adapter.reset();
		resetActionHistory();
	}

	AmPolicyAdapter::Observation observation{};
	PolicyObservationTiming policy_timing{};
	policy_timing.observation_build_timestamp = hrt_absolute_time();
	policy_timing.vehicle_local_position_timestamp = _position.timestamp;
	policy_timing.vehicle_local_position_timestamp_sample = _position.timestamp_sample;
	policy_timing.vehicle_attitude_timestamp = _attitude.timestamp;
	policy_timing.vehicle_attitude_timestamp_sample = _attitude.timestamp_sample;
	policy_timing.vehicle_angular_velocity_timestamp = _angular_velocity.timestamp;
	policy_timing.vehicle_angular_velocity_timestamp_sample = _angular_velocity.timestamp_sample;
	policy_timing.arm_joint_state_timestamp = _arm_joint_state.timestamp;
	policy_timing.arm_joint_state_timestamp_sample = _arm_joint_state.timestamp_sample;
	policy_timing.arm_joint_state_sequence = _arm_joint_state.sequence;
	policy_timing.arm_joint_state_position_wrap_mask = _arm_joint_state_position_wrap_mask;
	policy_timing.trajectory_setpoint_timestamp = _trajectory_setpoint.timestamp;
	policy_timing.offboard_control_mode_timestamp = _offboard_control_mode.timestamp;
	buildObservation(observation);

	AmPolicyAdapter::Action action{};
	policy_timing.policy_inference_start_timestamp = hrt_absolute_time();
	const bool inference_ok = _adapter.infer(policy_timing.policy_inference_start_timestamp, observation, action);
	policy_timing.policy_inference_finish_timestamp = hrt_absolute_time();
	policy_timing.takeoff_state = static_cast<uint8_t>(_takeoff.getTakeoffState());
	policy_timing.takeoff_ramped_speed_up = _takeoff_ramped_speed_up;
	policy_timing.takeoff_ramp_scale = takeoffRampOutputScale(policy_timing.takeoff_state, _takeoff_ramped_speed_up,
					  _takeoff_target_speed_up);

	if (inference_ok) {
		AmPolicyAdapter::Action executed_action{};
		policy_timing.policy_sequence = ++_policy_sequence;
		applyAction(observation, action, executed_action, mode, true, am_policy_degraded_flags, policy_timing);
		maybeLogPolicyDiagnostics(observation, action, executed_action);

		_am_offboard_using_external_setpoint = mode == ActiveMode::Offboard && am_offboard_using_external_setpoint;

		if (commit_policy_state) {
			updateActionHistory(executed_action);

		} else {
			_adapter.reset();
			resetActionHistory();
		}

	} else {
		_adapter.reset();
		publishStopSetpoint();
		resetCommandReference();
		resetActionHistory();
		_am_offboard_using_external_setpoint = false;
	}

	perf_end(_loop_perf);
}

int AmPosControl::task_spawn(int argc, char *argv[])
{
	AmPosControl *instance = new AmPosControl();

	if (instance) {
		_object.store(instance);
		_task_id = task_id_is_work_queue;

		if (instance->init()) {
			return PX4_OK;
		}

		PX4_ERR("init failed");
	}

	delete instance;
	_object.store(nullptr);
	_task_id = -1;
	return PX4_ERROR;
}

int AmPosControl::custom_command(int argc, char *argv[])
{
	return print_usage("unknown command");
}

int AmPosControl::print_status()
{
	const bool manual_control_available = manualControlAvailable();
	const bool vehicle_state_valid = vehicleStateValid();
	const bool attitude_valid = attitudeValid();
	const bool angular_velocity_valid = angularVelocityValid();
	const bool arm_state_valid = armStateValid();
	const bool trajectory_setpoint_valid = trajectorySetpointValid();
	const bool am_position_available = manual_control_available && arm_state_valid;
	const bool am_offboard_available = arm_state_valid;
	_offboard_control_mode_sub.update(&_offboard_control_mode);
	const bool offboard_control_mode_fresh = offboardControlModeFresh();
	const bool offboard_control_mode_supported = offboardControlModeSupported();
	const bool offboard_control_mode_valid = offboardControlModeValid();

	const bool manual_control_required = activeMode() != ActiveMode::Offboard;
	PX4_INFO("manual_control_required: %s, manual_control_available: %s",
		 manual_control_required ? "yes" : "no",
		 manual_control_available ? "yes" : "no");
	PX4_INFO("vehicle_state_valid: %s, attitude_valid: %s, angular_velocity_valid: %s",
		 vehicle_state_valid ? "yes" : "no",
		 attitude_valid ? "yes" : "no",
		 angular_velocity_valid ? "yes" : "no");
	PX4_INFO("offboard_control_mode_fresh: %s, supported: %s, valid: %s",
		 offboard_control_mode_fresh ? "yes" : "no",
		 offboard_control_mode_supported ? "yes" : "no",
		 offboard_control_mode_valid ? "yes" : "no");
	PX4_INFO("arm_state_valid: %s, trajectory_setpoint_valid: %s",
		 arm_state_valid ? "yes" : "no",
		 trajectory_setpoint_valid ? "yes" : "no");
	PX4_INFO("am_position_available: %s, am_offboard_available: %s",
		 am_position_available ? "yes" : "no",
		 am_offboard_available ? "yes" : "no");

	return 0;
}

int AmPosControl::print_usage(const char *reason)
{
	if (reason) {
		PX4_ERR("%s", reason);
	}

	PRINT_MODULE_DESCRIPTION(
		R"DESCR_STR(
### Description
AM Position direct-actuator controller.

This module executes the internal AM Position and AM Offboard modes. The
manual mode consumes FlightModeManager position-style setpoints while the
offboard mode consumes external trajectory setpoints, and both route direct
motor commands for a multicopter-with-arm setup.
)DESCR_STR");

	PRINT_MODULE_USAGE_NAME("am_pos_control", "controller");
	PRINT_MODULE_USAGE_COMMAND("start");
	PRINT_MODULE_USAGE_DEFAULT_COMMANDS();
	return 0;
}

extern "C" __EXPORT int am_pos_control_main(int argc, char *argv[])
{
	return AmPosControl::main(argc, argv);
}
