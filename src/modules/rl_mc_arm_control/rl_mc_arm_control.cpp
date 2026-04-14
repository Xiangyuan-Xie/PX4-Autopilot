/****************************************************************************
 *
 * RL multicopter arm direct actuator controller.
 *
 ****************************************************************************/

#include "rl_mc_arm_control.hpp"

#include <cmath>
#include <cstring>

namespace
{
constexpr char kModeName[] = "RL Arm Control";
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

RlMcArmControl::RlMcArmControl() :
	ModuleParams(nullptr),
	WorkItem(MODULE_NAME, px4::wq_configurations::nav_and_controllers),
	_loop_perf(perf_alloc(PC_ELAPSED, MODULE_NAME": cycle"))
{
	resetState();
}

RlMcArmControl::~RlMcArmControl()
{
	perf_free(_loop_perf);
}

bool RlMcArmControl::init()
{
	if (!_angular_velocity_sub.registerCallback()) {
		PX4_ERR("callback registration failed");
		return false;
	}

	return _adapter.init();
}

void RlMcArmControl::resetState()
{
	_trajectory_setpoint = {};
	_position = {};
	_attitude = {};
	_angular_velocity = {};
	_arm_joint_state = {};
	_prev_action = {{0.f, 0.f, 0.f, 0.f}};
	_root_pos_w = matrix::Vector3f();
	_root_quat_w = matrix::Quatf();
	_root_lin_vel_w = matrix::Vector3f();
	_root_lin_vel_b = matrix::Vector3f();
	_root_ang_vel_b = matrix::Vector3f();
	_heading_w = 0.0f;
	resetHoverLock();
	resetManualReferenceState();
	_adapter.reset();
}

void RlMcArmControl::resetHoverLock()
{
	_hover_lock_active = false;
	_hover_lock_pos_w = matrix::Vector3f();
	_hover_lock_yaw_w = 0.0f;
	_current_cmd_ref = {};
}

void RlMcArmControl::resetManualReferenceState()
{
	_manual_reference.reset();
	_last_manual_control_valid = false;
	_last_manual_control_nonzero = false;
	_last_manual_hold_xy = false;
	_last_manual_hold_z = false;
	_last_manual_hold_yaw = false;
	_last_raw_roll = 0.0f;
	_last_raw_pitch = 0.0f;
	_last_raw_yaw = 0.0f;
	_last_raw_throttle = 0.0f;
	_last_processed_roll = 0.0f;
	_last_processed_pitch = 0.0f;
	_last_processed_yaw = 0.0f;
	_last_processed_throttle = 0.0f;
	_last_desired_pos_w.zero();
	_last_desired_lin_vel_b.zero();
	_last_desired_ang_vel_b.zero();
}

bool RlMcArmControl::anyAxisActive(const std::array<bool, 3> &axes) const
{
	return axes[0] || axes[1] || axes[2];
}

void RlMcArmControl::registerMode()
{
	register_ext_component_request_s request{};
	request.timestamp = hrt_absolute_time();
	strncpy(request.name, kModeName, sizeof(request.name) - 1);
	request.request_id = _mode_request_id;
	request.px4_ros2_api_version = 1;
	request.register_arming_check = true;
	request.register_mode = true;
	_register_ext_component_request_pub.publish(request);
}

void RlMcArmControl::unregisterMode()
{
	if (_mode_id < 0) {
		return;
	}

	unregister_ext_component_s request{};
	request.timestamp = hrt_absolute_time();
	strncpy(request.name, kModeName, sizeof(request.name) - 1);
	request.arming_check_id = _arming_check_id;
	request.mode_id = _mode_id;
	_unregister_ext_component_pub.publish(request);
}

void RlMcArmControl::configureMode(int8_t mode_id)
{
	vehicle_control_mode_s config{};
	config.timestamp = hrt_absolute_time();
	config.source_id = mode_id;
	config.flag_multicopter_position_control_enabled = false;
	config.flag_control_manual_enabled = true;
	config.flag_control_offboard_enabled = false;
	config.flag_control_position_enabled = false;
	config.flag_control_velocity_enabled = false;
	config.flag_control_altitude_enabled = false;
	config.flag_control_climb_rate_enabled = false;
	config.flag_control_acceleration_enabled = false;
	config.flag_control_attitude_enabled = false;
	config.flag_control_rates_enabled = false;
	config.flag_control_allocation_enabled = false;
	config.flag_control_termination_enabled = true;
	_config_control_setpoints_pub.publish(config);
}

void RlMcArmControl::replyToArmingCheck(int8_t request_id)
{
	arming_check_reply_s reply{};
	reply.timestamp = hrt_absolute_time();
	reply.request_id = request_id;
	reply.registration_id = _arming_check_id;
	reply.health_component_index = reply.HEALTH_COMPONENT_INDEX_NONE;
	reply.num_events = 0;
	reply.can_arm_and_run = true;
	reply.mode_req_angular_velocity = false;
	reply.mode_req_local_position = true;
	reply.mode_req_attitude = false;
	reply.mode_req_local_alt = false;
	reply.mode_req_manual_control = true;

	_sticks.checkAndUpdateStickInputs();
	const bool manual_input_valid = _sticks.isAvailable();

	if (!manual_input_valid) {
		reply.can_arm_and_run = false;
	}

	_arming_check_reply_pub.publish(reply);
}

void RlMcArmControl::checkModeRegistration()
{
	register_ext_component_reply_s reply{};
	int tries = reply.ORB_QUEUE_LENGTH;

	while (_register_ext_component_reply_sub.update(&reply) && --tries >= 0) {
		if (reply.request_id != _mode_request_id) {
			continue;
		}

		if (reply.success) {
			_arming_check_id = reply.arming_check_id;
			_mode_id = reply.mode_id;
			configureMode(_mode_id);
			PX4_INFO("%s registered, mode id: %d", kModeName, _mode_id);

		} else {
			PX4_ERR("%s registration failed", kModeName);
		}

		break;
	}
}

void RlMcArmControl::generateManualTargets(matrix::Vector3f &desired_lin_vel_b,
	matrix::Vector3f &desired_ang_vel_b, matrix::Vector3f &desired_pos_w, matrix::Quatf &desired_quat_w,
	float dt_s)
{
	_sticks.checkAndUpdateStickInputs();
	_last_manual_control_valid = _sticks.isAvailable();
	_last_raw_roll = _sticks.getRoll();
	_last_raw_pitch = _sticks.getPitch();
	_last_raw_yaw = _sticks.getYaw();
	_last_raw_throttle = _sticks.getThrottleZeroCentered();

	RlMcArmControlManualReference::Params params{};
	params.vel_manual = _param_mapc_vel_manual.get();
	params.vel_side = _param_mapc_vel_side.get();
	params.vel_back = _param_mapc_vel_back.get();
	params.acc_hor = _param_mapc_acc_hor.get();
	params.jerk_max = _param_mapc_jerk_max.get();
	params.z_vel_up = _param_mapc_z_vel_up.get();
	params.z_vel_down = _param_mapc_z_vel_dn.get();
	params.man_y_max_rad = math::radians(_param_mapc_man_y_max.get());
	params.man_y_tau = _param_mapc_man_y_tau.get();
	params.man_deadzone = _param_mapc_man_dz.get();
	params.hold_max_xy = _param_mapc_hold_max_xy.get();
	params.hold_max_z = _param_mapc_hold_max_z.get();

	RlMcArmControlManualReference::Input input{};
	input.dt_s = dt_s;
	input.sticks_available = _last_manual_control_valid;
	input.roll = _last_raw_roll;
	input.pitch = _last_raw_pitch;
	input.yaw = _last_raw_yaw;
	input.throttle = _last_raw_throttle;
	input.root_pos_w = _root_pos_w;
	input.root_quat_w = _root_quat_w;
	input.root_lin_vel_w = _root_lin_vel_w;
	input.heading_w = _heading_w;
	input.position_z_ned = _position.z;
	input.velocity_z_ned = _position.vz;

	const RlMcArmControlManualReference::Output output = _manual_reference.update(params, input);

	desired_lin_vel_b = output.desired_lin_vel_b;
	desired_ang_vel_b = output.desired_ang_vel_b;
	desired_pos_w = output.desired_pos_w;
	desired_quat_w = output.desired_quat_w;

	_last_processed_roll = output.processed_roll;
	_last_processed_pitch = output.processed_pitch;
	_last_processed_yaw = output.processed_yaw;
	_last_processed_throttle = output.processed_throttle;
	_last_manual_control_nonzero = fabsf(output.processed_roll) > FLT_EPSILON
				       || fabsf(output.processed_pitch) > FLT_EPSILON
				       || fabsf(output.processed_yaw) > FLT_EPSILON
				       || fabsf(output.processed_throttle) > FLT_EPSILON;
	_last_manual_hold_xy = output.hold_xy_active;
	_last_manual_hold_z = output.hold_z_active;
	_last_manual_hold_yaw = output.hold_yaw_active;
	_last_desired_pos_w = desired_pos_w;
	_last_desired_lin_vel_b = desired_lin_vel_b;
	_last_desired_ang_vel_b = desired_ang_vel_b;
}

void RlMcArmControl::updateTargets(float dt_s)
{
	matrix::Vector3f desired_lin_vel_b{};
	matrix::Vector3f desired_ang_vel_b{};
	matrix::Vector3f desired_pos_w{_root_pos_w};
	matrix::Quatf desired_quat_w{yawOnlyQuat(_heading_w)};
	desired_lin_vel_b.zero();
	desired_ang_vel_b.zero();

	generateManualTargets(desired_lin_vel_b, desired_ang_vel_b, desired_pos_w, desired_quat_w, dt_s);

	std::array<bool, 3> lin_active{{
		fabsf(desired_lin_vel_b(0)) > kCmdZeroEps,
		fabsf(desired_lin_vel_b(1)) > kCmdZeroEps,
		fabsf(desired_lin_vel_b(2)) > kCmdZeroEps,
	}};

	std::array<bool, 3> ang_active{{
		false,
		false,
		fabsf(desired_ang_vel_b(2)) > kCmdZeroEps,
	}};

	_current_cmd_ref.desired_lin_vel_b = desired_lin_vel_b;
	_current_cmd_ref.desired_ang_vel_b = desired_ang_vel_b;
	_current_cmd_ref.desired_pos_w = desired_pos_w;
	_current_cmd_ref.desired_quat_w = desired_quat_w;
	_current_cmd_ref.lin_cmd_active = lin_active;
	_current_cmd_ref.ang_cmd_active = ang_active;
	_current_cmd_ref.has_lin_vel_cmd = anyAxisActive(lin_active);
	_current_cmd_ref.has_ang_vel_cmd = anyAxisActive(ang_active);
}

bool RlMcArmControl::updateVehicleState(float &dt_s)
{
	if (!_angular_velocity_sub.update(&_angular_velocity)) {
		return false;
	}

	dt_s = math::constrain((_angular_velocity.timestamp_sample - _last_run) * 1e-6f, 0.0002f, 0.02f);
	_last_run = _angular_velocity.timestamp_sample;

	_attitude_sub.update(&_attitude);
	_position_sub.update(&_position);
	_arm_joint_state_sub.update(&_arm_joint_state);
	if (!PX4_ISFINITE(_position.x) || !PX4_ISFINITE(_position.y) || !PX4_ISFINITE(_position.z)) {
		return false;
	}

	updateConvertedState();
	return true;
}

void RlMcArmControl::updateConvertedState()
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

void RlMcArmControl::buildObservation(RlToolsAdapter::Observation &observation)
{
	observation.fill(0.0f);

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

void RlMcArmControl::updateActionHistory(const RlToolsAdapter::Action &action)
{
	for (int i = 0; i < kActionDim; ++i) {
		_prev_action[i] = action[i];
	}
}

void RlMcArmControl::applyAction(const RlToolsAdapter::Action &action)
{
	actuator_motors_s actuator_motors{};
	actuator_motors.timestamp = hrt_absolute_time();

	for (int i = 0; i < kActionDim; ++i) {
		const float mapped = 1.0f / (1.0f + expf(-2.0f * action[i]));
		actuator_motors.control[i] = math::constrain(mapped, 0.0f, 1.0f);
	}

	for (int i = kActionDim; i < kMotorControlDim; ++i) {
		actuator_motors.control[i] = NAN;
	}

	actuator_motors.reversible_flags = 0;
	_actuator_motors_pub.publish(actuator_motors);
}

void RlMcArmControl::Run()
{
	if (should_exit()) {
		_angular_velocity_sub.unregisterCallback();
		unregisterMode();
		exit_and_cleanup();
		return;
	}

	if (!_sent_mode_registration) {
		registerMode();
		_sent_mode_registration = true;
		return;
	}

	if (_mode_id == -1 || _arming_check_id == -1) {
		checkModeRegistration();
		return;
	}

	perf_begin(_loop_perf);

	if (_arming_check_id >= 0 && _arming_check_request_sub.updated()) {
		arming_check_request_s arming_check_request{};
		_arming_check_request_sub.copy(&arming_check_request);
		replyToArmingCheck(arming_check_request.request_id);
	}

	vehicle_status_s vehicle_status{};
	const bool was_using_rl_mode = _use_rl_mode;
	if (_vehicle_status_sub.updated()) {
		_vehicle_status_sub.copy(&vehicle_status);
		_use_rl_mode = vehicle_status.nav_state == _mode_id;
	}

	if (_parameter_update_sub.updated()) {
		parameter_update_s param_update{};
		_parameter_update_sub.copy(&param_update);
		updateParams();
		if (_mode_id >= 0) {
			configureMode(_mode_id);
		}
	}

	if (_use_rl_mode && !was_using_rl_mode) {
		resetHoverLock();
		resetManualReferenceState();
		_adapter.reset();
	}

	if (!_use_rl_mode) {
		perf_end(_loop_perf);
		return;
	}

	float dt_s = 0.0f;
	if (!updateVehicleState(dt_s)) {
		perf_end(_loop_perf);
		return;
	}

	updateTargets(dt_s);

	RlToolsAdapter::Observation observation{};
	buildObservation(observation);

	RlToolsAdapter::Action action{};
	if (_adapter.infer(hrt_absolute_time(), observation, action)) {
		applyAction(action);
		updateActionHistory(action);
	}

	perf_end(_loop_perf);
}

int RlMcArmControl::task_spawn(int argc, char *argv[])
{
	RlMcArmControl *instance = new RlMcArmControl();

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

int RlMcArmControl::custom_command(int argc, char *argv[])
{
	return print_usage("unknown command");
}

int RlMcArmControl::print_status()
{
	if (_mode_id == -1) {
		PX4_INFO("%s registration pending", kModeName);

	} else {
		PX4_INFO("%s registered, mode id: %d, arming check id: %d", kModeName, _mode_id, _arming_check_id);
	}

	return 0;
}

int RlMcArmControl::print_usage(const char *reason)
{
	if (reason) {
		PX4_ERR("%s", reason);
	}

	PRINT_MODULE_DESCRIPTION(
		R"DESCR_STR(
### Description
RL multicopter arm direct-actuator controller.

This module registers a selectable external-mode slot from inside PX4 and
publishes direct motor commands for a multicopter-with-arm setup.
)DESCR_STR");

	PRINT_MODULE_USAGE_NAME("rl_mc_arm_control", "controller");
	PRINT_MODULE_USAGE_COMMAND("start");
	PRINT_MODULE_USAGE_DEFAULT_COMMANDS();
	return 0;
}

extern "C" __EXPORT int rl_mc_arm_control_main(int argc, char *argv[])
{
	return RlMcArmControl::main(argc, argv);
}
