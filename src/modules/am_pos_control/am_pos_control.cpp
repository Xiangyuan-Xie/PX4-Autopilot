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

	return _adapter.init();
}

void AmPosControl::resetCommandReference()
{
	_current_cmd_ref = {};
}

void AmPosControl::resetState()
{
	_trajectory_setpoint = {};
	_position = {};
	_attitude = {};
	_angular_velocity = {};
	_arm_joint_state = {};
	for (int i = 0; i < kActionDim; ++i) {
		_prev_action[i] = 0.0f;
	}
	_root_pos_w.zero();
	_root_quat_w = matrix::Quatf();
	_root_lin_vel_w.zero();
	_root_lin_vel_b.zero();
	_root_ang_vel_b.zero();
	_heading_w = 0.0f;
	resetCommandReference();
	_adapter.reset();
}

bool AmPosControl::anyAxisActive(const bool axes[3]) const
{
	return axes[0] || axes[1] || axes[2];
}

bool AmPosControl::updateVehicleState(float &dt_s, bool allow_am_test_degraded, uint32_t &degraded_flags)
{
	if (!_angular_velocity_sub.update(&_angular_velocity)) {
		return false;
	}

	dt_s = math::constrain((_angular_velocity.timestamp_sample - _last_run) * 1e-6f, 0.0002f, 0.02f);
	_last_run = _angular_velocity.timestamp_sample;

	_attitude_sub.update(&_attitude);
	_position_sub.update(&_position);
	_arm_joint_state_sub.update(&_arm_joint_state);

	const bool vehicle_state_valid = allow_am_test_degraded ?
					 vehicleStateValidForAmTest(_position, attitudeValid(), angularVelocityValid(),
							 hrt_absolute_time(), degraded_flags) :
					 vehicleStateValid();

	if (!vehicle_state_valid) {
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
		if (!PX4_ISFINITE(_arm_joint_state.arm_position[i]) || !PX4_ISFINITE(_arm_joint_state.arm_velocity[i])) {
			return false;
		}
	}

	return true;
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

	if (_offboard_control_mode.position) {
		return PX4_ISFINITE(_trajectory_setpoint.position[0]) || PX4_ISFINITE(_trajectory_setpoint.position[1])
		       || PX4_ISFINITE(_trajectory_setpoint.position[2]);
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
	const bool position_or_velocity = _offboard_control_mode.position || _offboard_control_mode.velocity;
	const bool unsupported = _offboard_control_mode.acceleration || _offboard_control_mode.attitude
				 || _offboard_control_mode.body_rate || _offboard_control_mode.thrust_and_torque
				 || _offboard_control_mode.direct_actuator;
	return position_or_velocity && !unsupported;
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

		if (vehicle_status.nav_state == vehicle_status_s::NAVIGATION_STATE_AM_TEST) {
			return ActiveMode::Test;
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
	status.arm_state_valid = armStateValid();
	status.trajectory_setpoint_valid = trajectorySetpointValid();
	status.offboard_control_mode_fresh = offboardControlModeFresh();
	status.offboard_control_mode_supported = offboardControlModeSupported();
	status.am_position_available = status.manual_control_available && status.arm_state_valid;
	status.am_offboard_available = status.arm_state_valid && status.offboard_control_mode_fresh
				       && status.offboard_control_mode_supported && offboardTrajectorySetpointValid();
	_status_pub.publish(status);
}

void AmPosControl::publishAmTestStatus(bool vehicle_state_valid, bool arm_state_valid, bool am_setpoint_valid, bool am_valid,
				       uint32_t failure_flags, uint32_t degraded_flags)
{
	am_test_status_s status{};
	status.timestamp = hrt_absolute_time();
	status.module_running = true;
	status.vehicle_state_valid = vehicle_state_valid;
	status.arm_state_valid = arm_state_valid;
	status.am_setpoint_valid = am_setpoint_valid;
	status.am_valid = am_valid;
	status.failure_flags = failure_flags;
	status.degraded_flags = degraded_flags;
	_am_test_status_pub.publish(status);
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

void AmPosControl::updateTargets(bool use_default_am_test_setpoint)
{
	_trajectory_setpoint_sub.update(&_trajectory_setpoint);

	if (use_default_am_test_setpoint) {
		uint32_t unused_degraded_flags = am_test_result_s::DEGRADED_NONE;
		fillDefaultAmTestSetpoint(_trajectory_setpoint, hrt_absolute_time(), _position, unused_degraded_flags);
	}

	matrix::Vector3f desired_pos_ned(_position.x, _position.y, _position.z);
	for (int i = 0; i < 3; ++i) {
		if (PX4_ISFINITE(_trajectory_setpoint.position[i])) {
			desired_pos_ned(i) = _trajectory_setpoint.position[i];
		}
	}

	matrix::Vector3f desired_vel_ned{};
	for (int i = 0; i < 3; ++i) {
		desired_vel_ned(i) = PX4_ISFINITE(_trajectory_setpoint.velocity[i]) ? _trajectory_setpoint.velocity[i] : 0.0f;
	}

	const matrix::Vector3f desired_vel_w = positionNedToEnu(desired_vel_ned);
	const matrix::Vector3f desired_lin_vel_b = _root_quat_w.inversed().rotateVector(desired_vel_w);
	matrix::Vector3f desired_ang_vel_b{};
	desired_ang_vel_b.zero();

	if (PX4_ISFINITE(_trajectory_setpoint.yawspeed)) {
		// `trajectory_setpoint` is expressed in NED, while the policy uses FLU/ENU.
		desired_ang_vel_b(2) = -_trajectory_setpoint.yawspeed;
	}

	const float desired_yaw_w = PX4_ISFINITE(_trajectory_setpoint.yaw) ? yawNedToEnu(_trajectory_setpoint.yaw) : _heading_w;
	const matrix::Quatf desired_quat_w = yawOnlyQuat(desired_yaw_w);
	const matrix::Vector3f desired_pos_w = positionNedToEnu(desired_pos_ned);

	const bool lin_active[3]{
		PX4_ISFINITE(_trajectory_setpoint.velocity[0]) && fabsf(desired_lin_vel_b(0)) > kCmdZeroEps,
		PX4_ISFINITE(_trajectory_setpoint.velocity[1]) && fabsf(desired_lin_vel_b(1)) > kCmdZeroEps,
		PX4_ISFINITE(_trajectory_setpoint.velocity[2]) && fabsf(desired_lin_vel_b(2)) > kCmdZeroEps,
	};

	const bool ang_active[3]{
		false,
		false,
		PX4_ISFINITE(_trajectory_setpoint.yawspeed) && fabsf(desired_ang_vel_b(2)) > kCmdZeroEps,
	};

	_current_cmd_ref.desired_lin_vel_b = desired_lin_vel_b;
	_current_cmd_ref.desired_ang_vel_b = desired_ang_vel_b;
	_current_cmd_ref.desired_pos_w = desired_pos_w;
	_current_cmd_ref.desired_quat_w = desired_quat_w;
	for (int i = 0; i < 3; ++i) {
		_current_cmd_ref.lin_cmd_active[i] = lin_active[i];
		_current_cmd_ref.ang_cmd_active[i] = ang_active[i];
	}
	_current_cmd_ref.has_lin_vel_cmd = anyAxisActive(lin_active);
	_current_cmd_ref.has_ang_vel_cmd = anyAxisActive(ang_active);
}

void AmPosControl::buildObservation(RlToolsAdapter::Observation &observation)
{
	for (int i = 0; i < RlToolsAdapter::ObservationDim; ++i) {
		observation[i] = 0.0f;
	}

	const matrix::Vector3f &root_pos_w = _root_pos_w;
	const matrix::Quatf &root_quat_w = _root_quat_w;
	const matrix::Vector3f &lin_vel_b = _root_lin_vel_b;
	const matrix::Vector3f &ang_vel_b = _root_ang_vel_b;
	const matrix::Vector3f pos_err_b = root_quat_w.inversed().rotateVector(_current_cmd_ref.desired_pos_w - root_pos_w);

	matrix::Vector3f gated_pos_err_b(pos_err_b);
	for (int i = 0; i < 3; ++i) {
		if (_current_cmd_ref.lin_cmd_active[i]) {
			gated_pos_err_b(i) = 0.0f;
		}
	}

	const matrix::Quatf att_err_quat = root_quat_w.inversed() * _current_cmd_ref.desired_quat_w;
	matrix::Eulerf att_err_euler(att_err_quat);
	float roll_err = wrapToPi(att_err_euler.phi());
	float pitch_err = wrapToPi(att_err_euler.theta());
	float yaw_err = wrapToPi(att_err_euler.psi());

	if (_current_cmd_ref.ang_cmd_active[0]) {
		roll_err = 0.0f;
	}

	if (_current_cmd_ref.ang_cmd_active[1]) {
		pitch_err = 0.0f;
	}

	if (_current_cmd_ref.ang_cmd_active[2]) {
		yaw_err = 0.0f;
	}

	const matrix::Dcmf att_err_dcm(matrix::Quatf(matrix::Eulerf(roll_err, pitch_err, yaw_err)));
	const matrix::Vector3f projected_gravity_b = root_quat_w.inversed().rotateVector(kGravityEnu);
	const matrix::Vector3f lin_vel_err_b = _current_cmd_ref.desired_lin_vel_b - lin_vel_b;
	const matrix::Vector3f ang_vel_err_b = _current_cmd_ref.desired_ang_vel_b - ang_vel_b;

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
		observation[idx++] = _arm_joint_state.arm_position[i];
	}

	for (int i = 0; i < kArmJointDim; ++i) {
		observation[idx++] = _arm_joint_state.arm_velocity[i];
	}

	for (int i = 0; i < kActionDim; ++i) {
		observation[idx++] = _prev_action[i];
	}
}

void AmPosControl::updateActionHistory(const RlToolsAdapter::Action &action)
{
	for (int i = 0; i < kActionDim; ++i) {
		_prev_action[i] = action[i];
	}
}

void AmPosControl::maybeLogPolicyDiagnostics(const RlToolsAdapter::Observation &observation, const RlToolsAdapter::Action &action)
{
	if (_startup_diag_samples_remaining <= 0) {
		return;
	}

	const ActiveMode mode = activeMode();
	const char *mode_label = mode == ActiveMode::Offboard ? "AM Offboard" :
				 mode == ActiveMode::Test ? "AM Test" : "AM Position";

	float mapped_action[kActionDim]{};
	float mapped_sum = 0.0f;
	float mapped_min = INFINITY;
	float mapped_max = -INFINITY;

	for (int i = 0; i < kActionDim; ++i) {
		mapped_action[i] = math::constrain(1.0f / (1.0f + expf(-2.0f * action[i])), 0.0f, 1.0f);
		mapped_sum += mapped_action[i];
		mapped_min = math::min(mapped_min, mapped_action[i]);
		mapped_max = math::max(mapped_max, mapped_action[i]);
	}

	const float mapped_mean = mapped_sum / static_cast<float>(kActionDim);
	const float mapped_spread = mapped_max - mapped_min;

	PX4_INFO(
		"%s obs pos_err=(%.3f, %.3f, %.3f) grav_z=%.3f prev_action=(%.3f, %.3f, %.3f, %.3f)",
		mode_label,
		(double)observation[0], (double)observation[1], (double)observation[2], (double)observation[14],
		(double)observation[31], (double)observation[32], (double)observation[33], (double)observation[34]);
	PX4_INFO(
		"%s act raw=(%.3f, %.3f, %.3f, %.3f) mapped=(%.3f, %.3f, %.3f, %.3f) mean=%.3f spread=%.3f",
		mode_label,
		(double)action[0], (double)action[1], (double)action[2], (double)action[3],
		(double)mapped_action[0], (double)mapped_action[1], (double)mapped_action[2], (double)mapped_action[3],
		(double)mapped_mean, (double)mapped_spread);

	if (mapped_mean < kDiagLowMeanCommand
	    || (mapped_min < kDiagLowMinCommand && mapped_spread > kDiagWideSpread)) {
		PX4_WARN(
			"%s action distribution looks weak/imbalanced: mean=%.3f min=%.3f max=%.3f",
			mode_label,
			(double)mapped_mean, (double)mapped_min, (double)mapped_max);
	}

	--_startup_diag_samples_remaining;
}

void AmPosControl::publishPolicyObservation(const RlToolsAdapter::Observation &observation,
		const RlToolsAdapter::Action &action, const actuator_motors_s &actuator_motors, uint32_t degraded_flags)
{
	am_policy_observation_s policy_observation{};
	policy_observation.timestamp = actuator_motors.timestamp;
	policy_observation.timestamp_sample = _angular_velocity.timestamp_sample;
	policy_observation.failure_flags = 0;
	policy_observation.degraded_flags = degraded_flags;
	policy_observation.am_setpoint_timestamp = _trajectory_setpoint.timestamp;

	for (int i = 0; i < RlToolsAdapter::ObservationDim; ++i) {
		policy_observation.observation[i] = observation[i];
	}

	for (int i = 0; i < kActionDim; ++i) {
		policy_observation.raw_action[i] = action[i];
		policy_observation.mapped_action[i] = actuator_motors.control[i];
	}

	for (int i = 0; i < kMotorControlDim; ++i) {
		policy_observation.motor_control[i] = actuator_motors.control[i];
	}

	_policy_observation_pub.publish(policy_observation);
}

void AmPosControl::publishAmTestResult(uint32_t failure_flags, uint32_t degraded_flags)
{
	am_test_result_s result{};
	fillInvalidAmTestResult(result, hrt_absolute_time(), _angular_velocity.timestamp_sample, _trajectory_setpoint.timestamp,
				failure_flags, degraded_flags);
	_am_test_result_pub.publish(result);
}

void AmPosControl::publishAmTestResult(const RlToolsAdapter::Action &action, uint32_t degraded_flags)
{
	am_test_result_s result{};
	fillAmTestResultFromAction(result, hrt_absolute_time(), _angular_velocity.timestamp_sample, _trajectory_setpoint.timestamp,
				   action, degraded_flags);
	_am_test_result_pub.publish(result);
}

void AmPosControl::applyAction(const RlToolsAdapter::Observation &observation, const RlToolsAdapter::Action &action,
			       bool publish_outputs, uint32_t degraded_flags)
{
	actuator_motors_s actuator_motors{};
	actuator_motors.timestamp = hrt_absolute_time();
	actuator_motors.timestamp_sample = actuator_motors.timestamp;

	for (int i = 0; i < kActionDim; ++i) {
		const float mapped = 1.0f / (1.0f + expf(-2.0f * action[i]));
		actuator_motors.control[i] = math::constrain(mapped, 0.0f, 1.0f);
	}

	for (int i = kActionDim; i < kMotorControlDim; ++i) {
		actuator_motors.control[i] = NAN;
	}

	actuator_motors.reversible_flags = 0;
	publishPolicyObservation(observation, action, actuator_motors, degraded_flags);

	if (publish_outputs) {
		_actuator_motors_pub.publish(actuator_motors);
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
	publishStatus();

	vehicle_status_s vehicle_status{};
	const bool was_using_am_mode = _use_am_mode;
	if (_vehicle_status_sub.updated()) {
		_vehicle_status_sub.copy(&vehicle_status);
		_use_am_mode = vehicle_status.nav_state == vehicle_status_s::NAVIGATION_STATE_AM_POSITION
			       || vehicle_status.nav_state == vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD
			       || vehicle_status.nav_state == vehicle_status_s::NAVIGATION_STATE_AM_TEST;
	}

	const ActiveMode mode = activeMode();
	const bool am_test_mode = mode == ActiveMode::Test;

	if (_parameter_update_sub.updated()) {
		parameter_update_s param_update{};
		_parameter_update_sub.copy(&param_update);
		updateParams();
	}

	if (_use_am_mode && !was_using_am_mode) {
		resetCommandReference();
		_adapter.reset();
		_startup_diag_samples_remaining = kStartupDiagSamples;
	}

	if (!_use_am_mode) {
		if (was_using_am_mode) {
			publishStopSetpoint();
			resetCommandReference();
			_adapter.reset();
		}

		perf_end(_loop_perf);
		return;
	}

	if (mode == ActiveMode::Offboard) {
		_offboard_control_mode_sub.update(&_offboard_control_mode);

		if (!offboardControlModeValid()) {
			_adapter.reset();
			resetCommandReference();

			if (hrt_elapsed_time(&_last_offboard_diag) > 1_s) {
				_last_offboard_diag = hrt_absolute_time();
				PX4_WARN("AM Offboard waiting for supported offboard_control_mode");
			}

			publishStopSetpoint();
			perf_end(_loop_perf);
			return;
		}
	}

	uint32_t am_test_degraded_flags = am_test_result_s::DEGRADED_NONE;
	float dt_s = 0.0f;
	if (!updateVehicleState(dt_s, am_test_mode, am_test_degraded_flags)) {
		_adapter.reset();
		resetCommandReference();

		if (am_test_mode) {
			publishAmTestStatus(false, false, false, false, am_test_result_s::FAILURE_VEHICLE_STATE_INVALID,
					    am_test_degraded_flags);
			publishAmTestResult(am_test_result_s::FAILURE_VEHICLE_STATE_INVALID, am_test_degraded_flags);

		} else {
			publishStopSetpoint();
		}

		perf_end(_loop_perf);
		return;
	}

	if (!armStateValid()) {
		_adapter.reset();
		resetCommandReference();

		if (am_test_mode) {
			publishAmTestStatus(true, false, false, false, am_test_result_s::FAILURE_ARM_STATE_INVALID,
					    am_test_degraded_flags);
			publishAmTestResult(am_test_result_s::FAILURE_ARM_STATE_INVALID, am_test_degraded_flags);

		} else {
			publishStopSetpoint();
		}

		perf_end(_loop_perf);
		return;
	}

	_trajectory_setpoint_sub.update(&_trajectory_setpoint);

	const bool trajectory_setpoint_valid = mode == ActiveMode::Offboard ? offboardTrajectorySetpointValid() :
					       trajectorySetpointValid();
	bool use_default_am_test_setpoint = false;

	if (!trajectory_setpoint_valid) {
		if (hrt_elapsed_time(&_last_setpoint_diag) > 1_s) {
			_last_setpoint_diag = hrt_absolute_time();
			PX4_WARN("AM Position waiting for fresh trajectory_setpoint");
		}

		if (am_test_mode) {
			use_default_am_test_setpoint = true;
			am_test_degraded_flags |= am_test_result_s::DEGRADED_SETPOINT_DEFAULTED;

		} else {
			_adapter.reset();
			resetCommandReference();
			publishStopSetpoint();
			perf_end(_loop_perf);
			return;
		}
	}

	updateTargets(use_default_am_test_setpoint);

	RlToolsAdapter::Observation observation{};
	buildObservation(observation);

	RlToolsAdapter::Action action{};
	if (_adapter.infer(hrt_absolute_time(), observation, action)) {
		maybeLogPolicyDiagnostics(observation, action);
		applyAction(observation, action, !am_test_mode, am_test_degraded_flags);

		if (am_test_mode) {
			publishAmTestStatus(true, true, true, true, am_test_result_s::FAILURE_NONE, am_test_degraded_flags);
			publishAmTestResult(action, am_test_degraded_flags);
		}

		updateActionHistory(action);

	} else {
		if (am_test_mode) {
			publishAmTestStatus(true, true, true, false, am_test_result_s::FAILURE_AM_INFERENCE_FAILED,
					    am_test_degraded_flags);
			publishAmTestResult(am_test_result_s::FAILURE_AM_INFERENCE_FAILED, am_test_degraded_flags);

		} else {
			publishStopSetpoint();
		}

		resetCommandReference();
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
	const bool arm_state_valid = armStateValid();
	const bool trajectory_setpoint_valid = trajectorySetpointValid();
	_offboard_control_mode_sub.update(&_offboard_control_mode);

	const bool manual_control_required = activeMode() != ActiveMode::Offboard;
	PX4_INFO("manual_control_required: %s, manual_control_available: %s",
		 manual_control_required ? "yes" : "no",
		 manual_control_available ? "yes" : "no");
	PX4_INFO("offboard_control_mode_valid: %s", offboardControlModeValid() ? "yes" : "no");
	PX4_INFO("arm_state_valid: %s, trajectory_setpoint_valid: %s",
		 arm_state_valid ? "yes" : "no",
		 trajectory_setpoint_valid ? "yes" : "no");

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
