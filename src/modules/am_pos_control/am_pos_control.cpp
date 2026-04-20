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
constexpr char kManualModeName[] = "AM Position";
constexpr char kOffboardModeName[] = "Offboard";
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
	_loop_perf(perf_alloc(PC_ELAPSED, MODULE_NAME": cycle")),
	_manual_mode(kManualModeName, 77, false),
	_offboard_mode(kOffboardModeName, 78, true)
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

void AmPosControl::registerMode(RegisteredMode &mode)
{
	register_ext_component_request_s request{};
	request.timestamp = hrt_absolute_time();
	strncpy(request.name, mode.name, sizeof(request.name) - 1);
	request.request_id = mode.request_id;
	request.px4_ros2_api_version = 1;
	request.register_arming_check = true;
	request.register_mode = true;

	if (mode.replace_offboard) {
		request.enable_replace_internal_mode = true;
		request.replace_internal_mode = vehicle_status_s::NAVIGATION_STATE_OFFBOARD;
	}

	_register_ext_component_request_pub.publish(request);
}

void AmPosControl::unregisterMode(RegisteredMode &mode)
{
	if (!mode.registered()) {
		return;
	}

	unregister_ext_component_s request{};
	request.timestamp = hrt_absolute_time();
	strncpy(request.name, mode.name, sizeof(request.name) - 1);
	request.arming_check_id = mode.arming_check_id;
	request.mode_id = mode.mode_id;
	_unregister_ext_component_pub.publish(request);

	mode.arming_check_id = -1;
	mode.mode_id = -1;
	mode.registration_requested = false;
}

void AmPosControl::configureMode(const RegisteredMode &mode)
{
	vehicle_control_mode_s config{};
	config.timestamp = hrt_absolute_time();
	config.source_id = mode.mode_id;
	config.flag_multicopter_position_control_enabled = false;
	config.flag_control_manual_enabled = !mode.replace_offboard;
	config.flag_control_offboard_enabled = mode.replace_offboard;
	config.flag_control_auto_enabled = false;
	config.flag_control_position_enabled = true;
	config.flag_control_velocity_enabled = true;
	config.flag_control_altitude_enabled = true;
	config.flag_control_climb_rate_enabled = true;
	config.flag_control_acceleration_enabled = false;
	config.flag_control_attitude_enabled = false;
	config.flag_control_rates_enabled = false;
	config.flag_control_allocation_enabled = false;
	config.flag_control_termination_enabled = true;
	_config_control_setpoints_pub.publish(config);
}

void AmPosControl::replyToArmingCheck(const RegisteredMode &mode, uint8_t request_id)
{
	arming_check_reply_s reply{};
	reply.timestamp = hrt_absolute_time();
	reply.request_id = request_id;
	reply.registration_id = mode.arming_check_id;
	reply.health_component_index = reply.HEALTH_COMPONENT_INDEX_NONE;
	reply.num_events = 0;
	reply.can_arm_and_run = true;
	reply.mode_req_angular_velocity = true;
	reply.mode_req_local_position = true;
	reply.mode_req_attitude = true;
	reply.mode_req_local_alt = true;
	reply.mode_req_manual_control = !mode.replace_offboard;

	_arm_joint_state_sub.update(&_arm_joint_state);
	_offboard_control_mode_sub.update(&_offboard_control_mode);
	const bool arm_state_valid = armStateValid();

	if (!mode.replace_offboard) {
		_sticks.checkAndUpdateStickInputs();
		const bool manual_control_available = _sticks.isAvailable();

		if (!manual_control_available && hrt_elapsed_time(&_last_manual_control_diag) > 1_s) {
			_last_manual_control_diag = hrt_absolute_time();
			PX4_WARN("AM Position waiting for manual control input");
		}
	}

	if (mode.replace_offboard && !offboardControlModeValid()) {
		reply.can_arm_and_run = false;

		if (hrt_elapsed_time(&_last_offboard_diag) > 1_s) {
			_last_offboard_diag = hrt_absolute_time();
			PX4_WARN("AM Position Offboard requires fresh supported offboard_control_mode");
		}
	}

	if (!arm_state_valid) {
		reply.can_arm_and_run = false;

		if (hrt_elapsed_time(&_last_arm_state_diag) > 1_s) {
			_last_arm_state_diag = hrt_absolute_time();
			PX4_WARN("AM Position cannot run: arm_joint_state invalid or stale");
		}
	}

	_arming_check_reply_pub.publish(reply);
}

void AmPosControl::checkModeRegistration()
{
	register_ext_component_reply_s reply{};
	int tries = reply.ORB_QUEUE_LENGTH;

	while (_register_ext_component_reply_sub.update(&reply) && --tries >= 0) {
		RegisteredMode *mode = nullptr;

		if (reply.request_id == _manual_mode.request_id) {
			mode = &_manual_mode;

		} else if (reply.request_id == _offboard_mode.request_id) {
			mode = &_offboard_mode;
		}

		if (!mode) {
			continue;
		}

		if (reply.success) {
			mode->arming_check_id = reply.arming_check_id;
			mode->mode_id = reply.mode_id;
			configureMode(*mode);
			PX4_INFO("%s registered, mode id: %d", mode->name, mode->mode_id);

		} else {
			PX4_ERR("%s registration failed", mode->name);
		}
	}
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
	const hrt_abstime now = hrt_absolute_time();

	if ((_position.timestamp == 0) || (now > _position.timestamp + kStateTimeout)) {
		return false;
	}

	const bool xy_valid = _position.xy_valid && PX4_ISFINITE(_position.x) && PX4_ISFINITE(_position.y);
	const bool z_valid = _position.z_valid && PX4_ISFINITE(_position.z);
	const bool v_xy_valid = _position.v_xy_valid && PX4_ISFINITE(_position.vx) && PX4_ISFINITE(_position.vy);
	const bool v_z_valid = _position.v_z_valid && PX4_ISFINITE(_position.vz);
	const bool heading_valid = PX4_ISFINITE(_position.heading);

	return xy_valid && z_valid && v_xy_valid && v_z_valid && heading_valid && attitudeValid() && angularVelocityValid();
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
	const hrt_abstime now = hrt_absolute_time();

	if ((_trajectory_setpoint.timestamp == 0) || (now > _trajectory_setpoint.timestamp + kTrajectorySetpointTimeout)) {
		return false;
	}

	return PX4_ISFINITE(_trajectory_setpoint.position[0]) || PX4_ISFINITE(_trajectory_setpoint.position[1])
	       || PX4_ISFINITE(_trajectory_setpoint.position[2]) || PX4_ISFINITE(_trajectory_setpoint.velocity[0])
	       || PX4_ISFINITE(_trajectory_setpoint.velocity[1]) || PX4_ISFINITE(_trajectory_setpoint.velocity[2])
	       || PX4_ISFINITE(_trajectory_setpoint.yaw) || PX4_ISFINITE(_trajectory_setpoint.yawspeed);
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

AmPosControl::ActiveMode AmPosControl::activeMode()
{
	vehicle_status_s vehicle_status{};

	if (_vehicle_status_sub.copy(&vehicle_status)) {
		if (_manual_mode.registered() && vehicle_status.nav_state == _manual_mode.mode_id) {
			return ActiveMode::Manual;
		}

		if (_offboard_mode.registered() && vehicle_status.nav_state == _offboard_mode.mode_id) {
			return ActiveMode::Offboard;
		}
	}

	return ActiveMode::None;
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
	_trajectory_setpoint_sub.update(&_trajectory_setpoint);

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

void AmPosControl::applyAction(const RlToolsAdapter::Action &action)
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

void AmPosControl::Run()
{
	if (should_exit()) {
		_angular_velocity_sub.unregisterCallback();
		unregisterMode(_manual_mode);
		unregisterMode(_offboard_mode);
		exit_and_cleanup();
		return;
	}

	if (!_manual_mode.registration_requested) {
		registerMode(_manual_mode);
		_manual_mode.registration_requested = true;
		return;
	}

	if (_param_ampc_offb_en.get() && !_offboard_mode.registration_requested) {
		registerMode(_offboard_mode);
		_offboard_mode.registration_requested = true;
		return;
	}

	perf_begin(_loop_perf);

	checkModeRegistration();

	if (_arming_check_request_sub.updated()) {
		arming_check_request_s arming_check_request{};
		_arming_check_request_sub.copy(&arming_check_request);

		if (_manual_mode.registered()) {
			replyToArmingCheck(_manual_mode, arming_check_request.request_id);
		}

		if (_offboard_mode.registered()) {
			replyToArmingCheck(_offboard_mode, arming_check_request.request_id);
		}
	}

	vehicle_status_s vehicle_status{};
	const bool was_using_am_mode = _use_am_mode;
	if (_vehicle_status_sub.updated()) {
		_vehicle_status_sub.copy(&vehicle_status);
		_use_am_mode = (_manual_mode.registered() && vehicle_status.nav_state == _manual_mode.mode_id)
			       || (_offboard_mode.registered() && vehicle_status.nav_state == _offboard_mode.mode_id);
	}

	if (_parameter_update_sub.updated()) {
		parameter_update_s param_update{};
		_parameter_update_sub.copy(&param_update);
		updateParams();

		if (_manual_mode.registered()) {
			configureMode(_manual_mode);
		}

		if (_offboard_mode.registered()) {
			configureMode(_offboard_mode);
		}
	}

	if (_use_am_mode && !was_using_am_mode) {
		resetCommandReference();
		_adapter.reset();
	}

	if (!_use_am_mode) {
		perf_end(_loop_perf);
		return;
	}

	if (activeMode() == ActiveMode::Offboard) {
		_offboard_control_mode_sub.update(&_offboard_control_mode);

		if (!offboardControlModeValid()) {
			_adapter.reset();
			resetCommandReference();

			if (hrt_elapsed_time(&_last_offboard_diag) > 1_s) {
				_last_offboard_diag = hrt_absolute_time();
				PX4_WARN("AM Position Offboard waiting for supported offboard_control_mode");
			}

			perf_end(_loop_perf);
			return;
		}
	}

	float dt_s = 0.0f;
	if (!updateVehicleState(dt_s)) {
		_adapter.reset();
		resetCommandReference();
		perf_end(_loop_perf);
		return;
	}

	if (!armStateValid()) {
		_adapter.reset();
		resetCommandReference();
		perf_end(_loop_perf);
		return;
	}

	_trajectory_setpoint_sub.update(&_trajectory_setpoint);

	if (!trajectorySetpointValid()) {
		_adapter.reset();
		resetCommandReference();

		if (hrt_elapsed_time(&_last_setpoint_diag) > 1_s) {
			_last_setpoint_diag = hrt_absolute_time();
			PX4_WARN("AM Position waiting for fresh trajectory_setpoint");
		}

		perf_end(_loop_perf);
		return;
	}

	updateTargets();

	RlToolsAdapter::Observation observation{};
	buildObservation(observation);

	RlToolsAdapter::Action action{};
	if (_adapter.infer(hrt_absolute_time(), observation, action)) {
		applyAction(action);
		updateActionHistory(action);
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
	_sticks.checkAndUpdateStickInputs();
	const bool manual_control_available = _sticks.isAvailable();
	const bool arm_state_valid = armStateValid();
	const bool trajectory_setpoint_valid = trajectorySetpointValid();
	_offboard_control_mode_sub.update(&_offboard_control_mode);

	if (!_manual_mode.registered()) {
		PX4_INFO("%s registration pending", _manual_mode.name);

	} else {
		PX4_INFO("%s registered, mode id: %d, arming check id: %d",
			 _manual_mode.name, _manual_mode.mode_id, _manual_mode.arming_check_id);
	}

	if (_param_ampc_offb_en.get()) {
		if (!_offboard_mode.registered()) {
			PX4_INFO("%s replacement registration pending", _offboard_mode.name);

		} else {
			PX4_INFO("%s replacement registered, mode id: %d, arming check id: %d",
				 _offboard_mode.name, _offboard_mode.mode_id, _offboard_mode.arming_check_id);
		}
	}

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

This module registers a selectable AM Position mode and can optionally
replace the internal Offboard mode. It consumes trajectory setpoints and
publishes direct motor commands for a multicopter-with-arm setup.
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
