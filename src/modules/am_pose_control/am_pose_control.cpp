/****************************************************************************
 *
 * AM Pose direct actuator controller.
 *
 ****************************************************************************/

#include "am_pose_control.hpp"
#include "../commander/ModeUtil/control_mode.hpp"

#include <cmath>
#include <cstring>

namespace
{
const matrix::Vector3f kGravityEnu{0.0f, 0.0f, -1.0f};
constexpr float kHalfSqrt2 = 0.7071067811865476f;

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
	return matrix::wrap_pi(M_PI_F / 2.0f - yaw_ned_rad);
}
}

AmPoseControl::AmPoseControl() :
	ModuleParams(nullptr),
	ScheduledWorkItem(MODULE_NAME, px4::wq_configurations::nav_and_controllers),
	_loop_perf(perf_alloc(PC_ELAPSED, MODULE_NAME": cycle")),
	_trajectory_reference_perf(perf_alloc(PC_ELAPSED, MODULE_NAME": trajectory reference")),
	_trajectory_command_perf(perf_alloc(PC_ELAPSED, MODULE_NAME": trajectory command"))
{
	resetState();
}

AmPoseControl::~AmPoseControl()
{
	perf_free(_loop_perf);
	perf_free(_trajectory_reference_perf);
	perf_free(_trajectory_command_perf);
}

bool AmPoseControl::init()
{
	if (!_angular_velocity_sub.registerCallback()) {
		PX4_ERR("callback registration failed");
		return false;
	}

	_takeoff_status_pub.advertise();
	_takeoff.setSpoolupTime(_param_com_spoolup_time.get());
	_takeoff.setTakeoffRampTime(_param_ampc_tko_ramp_t.get());

	_policy_loaded = _adapter.init();

	if (!_policy_loaded) {
		PX4_WARN("AM Pose policy blob unavailable; AM Pose remains disabled");
	}

	ScheduleOnInterval(WatchdogIntervalUs);
	return true;
}

void AmPoseControl::resetState()
{
	_trajectory_setpoint = {};
	_am_pose_trajectory = {};
	_offboard_control_mode = {};
	_received_offboard_controller_type = offboard_control_mode_s::CONTROLLER_TYPE_NATIVE;
	_received_offboard_control_mode_timestamp = 0;
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
	_handover_action_timestamp = 0;
	_handover_action_valid = false;
	_handover_action_consumed = false;
	_handover_entry_accepted = false;

	for (float &action : _handover_action) {
		action = 0.f;
	}

	_root_pos_w.zero();
	_root_quat_w = matrix::Quatf();
	_root_lin_vel_w.zero();
	_root_lin_vel_b.zero();
	_root_ang_vel_b.zero();
	_heading_w = 0.0f;

	for (float &arm_position : _policy_arm_position) {
		arm_position = 0.0f;
	}

	for (float &arm_velocity : _policy_arm_velocity) {
		arm_velocity = 0.0f;
	}

	_arm_joint_state_position_wrap_mask = 0;
	_policy_sequence = 0;
	_next_policy_deadline = 0;
	_takeoff_output_released = false;
	_policy_mode = ActiveMode::None;
	_blend_start = 0;
	_last_pose_reference_update = 0;
	_last_pose_target_valid = false;
	_manual_hold_anchor_active = false;
	_trajectory_message_valid = false;
	_active_reference_source = ReferenceSource::Hold;
	_active_trajectory_id = 0;
	_last_accepted_trajectory_id = 0;
	_active_trajectory_point_count = 0;
	_last_trajectory_message_timestamp = 0;
	_last_point_setpoint_timestamp = 0;
	_last_velocity_setpoint_timestamp = 0;
	_has_accepted_trajectory_id = false;
	_offboard_velocity_hold_active = false;
	_offboard_velocity_setpoint_lost = false;
	_active_trajectory_min_z_ned = NAN;
	_estimator_reset_counters_initialized = false;

	for (float &action : _blend_start_action) {
		action = 0.f;
	}

	_adapter.reset();
}

void AmPoseControl::publishTakeoffStatus()
{
	const uint8_t takeoff_state = static_cast<uint8_t>(_takeoff.getTakeoffState());

	if (takeoff_state != _takeoff_status_pub.get().takeoff_state) {
		_takeoff_status_pub.get().takeoff_state = takeoff_state;
		_takeoff_status_pub.get().tilt_limit = NAN;
		_takeoff_status_pub.get().timestamp = hrt_absolute_time();
		_takeoff_status_pub.update();
	}
}

bool AmPoseControl::updateTakeoffGate(ActiveMode mode, bool was_using_am_mode, float dt_s)
{
	_vehicle_control_mode_sub.update(&_vehicle_control_mode);
	_vehicle_land_detected_sub.update(&_vehicle_land_detected);
	_vehicle_constraints_sub.update(&_vehicle_constraints);

	bool want_takeoff = _vehicle_constraints.want_takeoff;
	float speed_up = PX4_ISFINITE(_vehicle_constraints.speed_up) ? _vehicle_constraints.speed_up :
			 _param_ampc_z_vel_up.get();

	if (mode == ActiveMode::Offboard) {
		if (_active_reference_source == ReferenceSource::Trajectory
		    && PX4_ISFINITE(_active_trajectory_min_z_ned) && PX4_ISFINITE(_position.z)) {
			want_takeoff = _active_trajectory_min_z_ned < _position.z;

		} else {
			want_takeoff = offboardSetpointWantsTakeoff(_offboard_control_mode, _trajectory_setpoint, _position,
					_position.timestamp_sample);
		}

		speed_up = _param_ampc_z_vel_up.get();
	}

	if (!_vehicle_control_mode.flag_armed || !_use_am_mode) {
		_takeoff_output_released = false;
		_handover_action_consumed = false;
		_handover_entry_accepted = false;
	}

	const bool landed_for_takeoff = _takeoff_output_released ? false : _vehicle_land_detected.landed;
	const bool skip_takeoff = shouldSkipTakeoffGateOnAmModeEntry(_use_am_mode, was_using_am_mode,
				  landed_for_takeoff);
	_takeoff.updateTakeoffState(_vehicle_control_mode.flag_armed, landed_for_takeoff,
				    want_takeoff, speed_up, skip_takeoff, _position.timestamp_sample);

	// Advance the shared takeoff state machine without feeding a ramped target into AM policy inputs.
	_takeoff.updateRamp(dt_s, speed_up);

	publishTakeoffStatus();

	const uint8_t takeoff_state = static_cast<uint8_t>(_takeoff.getTakeoffState());
	_takeoff_output_released = nextTakeoffOutputReleased(takeoff_state, _vehicle_control_mode.flag_armed, _use_am_mode,
					   _takeoff_output_released);

	if (_takeoff_output_released && _vehicle_control_mode.flag_armed && _use_am_mode) {
		_handover_entry_accepted = true;
	}

	return takeoffOutputGateActive(takeoff_state, _takeoff_output_released);
}

bool AmPoseControl::updateVehicleState(float &dt_s)
{
	dt_s = PolicyIntervalUs * 1e-6f;
	_attitude_sub.update(&_attitude);
	_position_sub.update(&_position);
	_arm_joint_state_sub.update(&_arm_joint_state);

	if (!vehicleStateValid()) {
		return false;
	}

	updateConvertedState();
	return true;
}

bool AmPoseControl::attitudeValid() const
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

bool AmPoseControl::angularVelocityValid() const
{
	const hrt_abstime now = hrt_absolute_time();

	if ((_angular_velocity.timestamp == 0) || (now > _angular_velocity.timestamp + kStateTimeout)) {
		return false;
	}

	return PX4_ISFINITE(_angular_velocity.xyz[0]) && PX4_ISFINITE(_angular_velocity.xyz[1])
	       && PX4_ISFINITE(_angular_velocity.xyz[2]);
}

bool AmPoseControl::vehicleStateValid() const
{
	return vehicleStateValidStrict(_position, attitudeValid(), angularVelocityValid(), hrt_absolute_time());
}

bool AmPoseControl::armStateValid() const
{
	const AmPosePolicyAdapter::ObservationContract contract = _adapter.observationContract();

	if (contract == AmPosePolicyAdapter::ObservationContract::Unsupported) {
		return false;
	}

	if (!AmPosePolicyAdapter::usesArmPosition(contract) && !AmPosePolicyAdapter::usesArmVelocity(contract)) {
		return true;
	}

	const hrt_abstime now = hrt_absolute_time();

	if ((_arm_joint_state.timestamp == 0) || (now > _arm_joint_state.timestamp + kArmStateTimeout)) {
		return false;
	}

	return armJointStateValuesValid(_arm_joint_state, AmPosePolicyAdapter::usesArmPosition(contract),
				       AmPosePolicyAdapter::usesArmVelocity(contract));
}

void AmPoseControl::updatePolicyArmJointState()
{
	_arm_joint_state_position_wrap_mask = 0;
	const AmPosePolicyAdapter::ObservationContract contract = _adapter.observationContract();
	const bool use_arm_position = AmPosePolicyAdapter::usesArmPosition(contract);
	const bool use_arm_velocity = AmPosePolicyAdapter::usesArmVelocity(contract);

	for (int i = 0; i < kPolicyArmJointDim; ++i) {
		_policy_arm_position[i] = 0.f;
		_policy_arm_velocity[i] = 0.f;

		if (use_arm_position) {
			_policy_arm_position[i] = prepareArmJointPositionForPolicy(_arm_joint_state.arm_position[i],
						  _arm_joint_state_position_wrap_mask, i);
		}

		if (use_arm_velocity) {
			_policy_arm_velocity[i] = _arm_joint_state.arm_velocity[i];
		}
	}
}

bool AmPoseControl::trajectorySetpointValid(ActiveMode mode) const
{
	if (mode == ActiveMode::Offboard) {
		// AM Pose Offboard has an internal hold reference, so it can be entered before
		// an external position command arrives. A fresh unsupported command is
		// still rejected rather than silently interpreting it as a pose command.
		return !offboardAmPoseInputRejected(_offboard_control_mode, _trajectory_setpoint,
						offboardControlModeFresh(), trajectorySetpointFresh());
	}

	return !trajectorySetpointLost() && trajectorySetpointHasLinearInput();
}

bool AmPoseControl::trajectorySetpointFresh() const
{
	const hrt_abstime now = hrt_absolute_time();
	const hrt_abstime timeout = static_cast<hrt_abstime>(math::max(_param_ampc_trj_timeout.get(), 0.0f) * 1_s);

	return trajectorySetpointFreshAt(_trajectory_setpoint, now, timeout);
}

bool AmPoseControl::trajectorySetpointLost() const
{
	const hrt_abstime now = hrt_absolute_time();
	const hrt_abstime loss_timeout = static_cast<hrt_abstime>(math::max(_param_ampc_trj_loss_t.get(), 0.0f) * 1_s);

	return trajectorySetpointLostAt(_trajectory_setpoint, now, loss_timeout);
}

bool AmPoseControl::trajectorySetpointHasLinearInput() const
{
	return trajectorySetpointHasLinearInput(_trajectory_setpoint);
}

bool AmPoseControl::offboardTrajectorySetpointValid() const
{
	return offboardAmPoseExternalSetpointUsable(_offboard_control_mode, _trajectory_setpoint,
						offboardControlModeValid(), trajectorySetpointFresh() && !trajectorySetpointLost());
}

bool AmPoseControl::offboardControlModeFresh() const
{
	const hrt_abstime now = hrt_absolute_time();
	const hrt_abstime timeout = static_cast<hrt_abstime>(_param_com_of_loss_t.get() * 1_s);
	return (_offboard_control_mode.timestamp != 0) && (now <= _offboard_control_mode.timestamp + timeout);
}

bool AmPoseControl::offboardControlModeSupported() const
{
	return offboardControlModeSupported(_offboard_control_mode);
}

bool AmPoseControl::offboardControlModeValid() const
{
	const bool controller_selected = _offboard_control_mode.controller_type
					 == offboard_control_mode_s::CONTROLLER_TYPE_AM_POSE
					 || mode_util::isAmPoseOffboardControlMode(_vehicle_control_mode);
	return offboardControlModeFresh() && controller_selected && offboardControlModeSupported();
}

void AmPoseControl::updateOffboardControlMode()
{
	offboard_control_mode_s incoming{};

	if (_offboard_control_mode_sub.update(&incoming)) {
		_received_offboard_controller_type = incoming.controller_type;
		_received_offboard_control_mode_timestamp = incoming.timestamp;

		if (!mode_util::isAmPoseOffboardControlMode(_vehicle_control_mode)
		    || incoming.controller_type == offboard_control_mode_s::CONTROLLER_TYPE_AM_POSE) {
			_offboard_control_mode = incoming;
		}
	}
}

bool AmPoseControl::manualControlAvailable()
{
	_sticks.checkAndUpdateStickInputs();
	return _sticks.isAvailable();
}

AmPoseControl::ActiveMode AmPoseControl::activeMode()
{
	vehicle_status_s vehicle_status{};

	if (_vehicle_status_sub.copy(&vehicle_status)) {
		if (vehicle_status.nav_state == vehicle_status_s::NAVIGATION_STATE_AM_POSE) {
			return ActiveMode::Manual;
		}

		if (vehicle_status.nav_state == vehicle_status_s::NAVIGATION_STATE_OFFBOARD
		    && mode_util::isAmPoseOffboardControlMode(_vehicle_control_mode)) {
			return ActiveMode::Offboard;
		}
	}

	return ActiveMode::None;
}

void AmPoseControl::publishStatus()
{
	updateOffboardControlMode();
	_arm_joint_state_sub.update(&_arm_joint_state);
	_trajectory_setpoint_sub.update(&_trajectory_setpoint);
	_am_pose_trajectory_sub.update(&_am_pose_trajectory);

	am_pose_control_status_s status{};
	const ActiveMode mode = activeMode();
	const hrt_abstime now = hrt_absolute_time();
	const bool handover_required = _vehicle_control_mode.flag_armed && !_vehicle_land_detected.landed;
	const bool fresh_handover = handoverActionFresh(now);
	status.timestamp = now;
	status.module_running = true;
	status.manual_control_available = manualControlAvailable();
	status.vehicle_state_valid = vehicleStateValid();
	status.attitude_valid = attitudeValid();
	status.angular_velocity_valid = angularVelocityValid();
	status.arm_state_valid = armStateValid();
	status.trajectory_setpoint_valid = trajectorySetpointValid(mode);
	status.offboard_control_mode_fresh = offboardControlModeFresh();
	status.offboard_control_mode_supported = offboardControlModeSupported();
	status.offboard_control_mode_valid = offboardControlModeValid();
	status.policy_loaded = _policy_loaded;
	status.policy_observation_dim = _adapter.observationDim();
	status.policy_uses_arm_position = _adapter.usesArmPosition();
	status.policy_uses_arm_velocity = _adapter.usesArmVelocity();
	status.handover_action_ready = handoverActionReady(handover_required, _handover_entry_accepted, fresh_handover);
	status.handover_action_consumed = _handover_action_consumed;
	status.handover_action_age_s = _handover_action_timestamp != 0 && now >= _handover_action_timestamp
				       ? (now - _handover_action_timestamp) * 1e-6f : NAN;
	status.am_pose_available = status.manual_control_available && status.arm_state_valid && _policy_loaded
				   && status.handover_action_ready;
	status.offboard_am_pose_available = status.arm_state_valid && _policy_loaded && status.handover_action_ready;
	status.active_offboard_controller = am_pose_control_status_s::OFFBOARD_CONTROLLER_INACTIVE;

	if (_vehicle_control_mode.flag_control_offboard_enabled) {
		status.active_offboard_controller = mode_util::isAmPoseOffboardControlMode(_vehicle_control_mode)
						    ? am_pose_control_status_s::OFFBOARD_CONTROLLER_AM_POSE
						    : am_pose_control_status_s::OFFBOARD_CONTROLLER_NATIVE;
		const uint8_t active_type = status.active_offboard_controller
					    == am_pose_control_status_s::OFFBOARD_CONTROLLER_AM_POSE
					    ? offboard_control_mode_s::CONTROLLER_TYPE_AM_POSE
					    : offboard_control_mode_s::CONTROLLER_TYPE_NATIVE;
		const hrt_abstime offboard_timeout = static_cast<hrt_abstime>(_param_com_of_loss_t.get() * 1_s);
		const bool received_selector_fresh = _received_offboard_control_mode_timestamp != 0
						     && now <= _received_offboard_control_mode_timestamp + offboard_timeout;
		status.offboard_selector_mismatch = received_selector_fresh
						    && _received_offboard_controller_type != active_type;
	}
	const hrt_abstime blend_duration = static_cast<hrt_abstime>(math::max(_param_ampc_sw_blend_t.get(), 0.f) * 1_s);
	status.policy_blend_active = _blend_start != 0 && hrt_absolute_time() < _blend_start + blend_duration;
	status.trajectory_message_valid = _trajectory_message_valid;
	status.velocity_setpoint_lost = _offboard_velocity_setpoint_lost;
	status.active_reference_source = static_cast<uint8_t>(_active_reference_source);

	if (_active_reference_source == ReferenceSource::Trajectory) {
		status.active_trajectory_id = _active_trajectory_id;
		status.trajectory_point_count = _active_trajectory_point_count;
		status.trajectory_segment_index = _pose_trajectory.waypointSegmentIndex(status.timestamp);
		status.trajectory_elapsed_s = _pose_trajectory.waypointElapsedTime(status.timestamp);
		status.trajectory_duration_s = _pose_trajectory.waypointDuration();
	}

	_status_pub.publish(status);
}

void AmPoseControl::updateConvertedState()
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

void AmPoseControl::resetPoseReference(hrt_abstime now)
{
	_pose_trajectory.reset(_root_pos_w, _heading_w, now);
	_last_pose_reference_update = now;
	_last_pose_target_valid = false;
	_manual_hold_anchor_active = false;
	_active_reference_source = ReferenceSource::Hold;
	_active_trajectory_id = 0;
	_active_trajectory_point_count = 0;
	_active_trajectory_min_z_ned = NAN;
	_trajectory_message_valid = false;
	_last_accepted_trajectory_id = 0;
	_has_accepted_trajectory_id = false;
	_offboard_velocity_hold_active = false;
	_offboard_velocity_setpoint_lost = false;
}

void AmPoseControl::processTrajectoryCommand(hrt_abstime now, float maximum_velocity, float maximum_yaw_rate)
{
	if (_am_pose_trajectory.timestamp == 0
	    || _am_pose_trajectory.timestamp == _last_trajectory_message_timestamp) {
		return;
	}

	_last_trajectory_message_timestamp = _am_pose_trajectory.timestamp;
	const hrt_abstime timeout = static_cast<hrt_abstime>(math::max(_param_ampc_trj_timeout.get(), 0.f) * 1_s);

	if (now > _am_pose_trajectory.timestamp + timeout) {
		_trajectory_message_valid = false;
		PX4_WARN("Rejected stale AM Pose trajectory");
		return;
	}

	if (_am_pose_trajectory.command == am_pose_trajectory_s::COMMAND_CANCEL) {
		if (_am_pose_trajectory.point_count != 0) {
			_trajectory_message_valid = false;
			PX4_WARN("AM Pose trajectory cancel must contain zero points");
			return;
		}

		_pose_trajectory.cancelWaypoints(now);
		_trajectory_message_valid = true;
		_active_reference_source = ReferenceSource::Hold;
		_active_trajectory_id = 0;
		_active_trajectory_point_count = 0;
		_active_trajectory_min_z_ned = NAN;
		_last_accepted_trajectory_id = 0;
		_has_accepted_trajectory_id = false;
		_last_pose_target_valid = false;
		_offboard_velocity_hold_active = false;
		_offboard_velocity_setpoint_lost = false;
		// Do not let the cached point setpoint immediately undo an explicit cancel.
		// A point publisher can take control again by publishing a newer sample.
		_last_point_setpoint_timestamp = _trajectory_setpoint.timestamp;
		return;
	}

	if (_am_pose_trajectory.command != am_pose_trajectory_s::COMMAND_REPLACE) {
		_trajectory_message_valid = false;
		PX4_WARN("Rejected AM Pose trajectory command %u", _am_pose_trajectory.command);
		return;
	}

	if (_has_accepted_trajectory_id
	    && _am_pose_trajectory.trajectory_id == _last_accepted_trajectory_id) {
		_trajectory_message_valid = true;
		return;
	}

	AmPoseTrajectory::Waypoint waypoints[AmPoseTrajectory::kMaximumWaypointCount] {};
	const int point_count = _am_pose_trajectory.point_count;

	if (point_count < 1 || point_count > AmPoseTrajectory::kMaximumWaypointCount) {
		_trajectory_message_valid = false;
		PX4_WARN("Rejected AM Pose trajectory with %d points", point_count);
		return;
	}

	for (int index = 0; index < point_count; ++index) {
		const am_pose_trajectory_waypoint_s &point = _am_pose_trajectory.points[index];
		AmPoseTrajectory::Waypoint &waypoint = waypoints[index];
		waypoint.position = positionNedToEnu(matrix::Vector3f{point.position});
		waypoint.velocity = positionNedToEnu(matrix::Vector3f{point.velocity});
		waypoint.acceleration = positionNedToEnu(matrix::Vector3f{point.acceleration});
		waypoint.yaw = yawNedToEnu(point.yaw);
		waypoint.yaw_rate = -point.yaw_rate;
		waypoint.yaw_acceleration = -point.yaw_acceleration;
	}

	const AmPoseTrajectory::WaypointLoadResult result = _pose_trajectory.loadWaypoints(
				waypoints, point_count, now, maximum_velocity, maximum_yaw_rate,
				!_am_pose_trajectory.continue_at_end);

	if (result != AmPoseTrajectory::WaypointLoadResult::Accepted) {
		_trajectory_message_valid = false;
		PX4_WARN("Rejected AM Pose trajectory (%u)", static_cast<unsigned>(result));
		return;
	}

	_trajectory_message_valid = true;
	_active_reference_source = ReferenceSource::Trajectory;
	_active_trajectory_id = _am_pose_trajectory.trajectory_id;
	_last_accepted_trajectory_id = _am_pose_trajectory.trajectory_id;
	_has_accepted_trajectory_id = true;
	_active_trajectory_point_count = static_cast<uint8_t>(point_count);
	_offboard_velocity_hold_active = false;
	_offboard_velocity_setpoint_lost = false;
	// A point setpoint cached before this atomic trajectory must not take over
	// when the trajectory reaches its endpoint. Only a newer sample may do so.
	_last_point_setpoint_timestamp = _trajectory_setpoint.timestamp;
	_active_trajectory_min_z_ned = _am_pose_trajectory.points[0].position[2];

	for (int index = 1; index < point_count; ++index) {
		_active_trajectory_min_z_ned = math::min(
						       _active_trajectory_min_z_ned, _am_pose_trajectory.points[index].position[2]);
	}

	_last_pose_target_valid = false;
}

void AmPoseControl::updatePoseReference(ActiveMode mode, hrt_abstime now)
{
	if (!_pose_trajectory.initialized()) {
		resetPoseReference(now);
	}

	const bool offboard_velocity_mode = mode == ActiveMode::Offboard
					    && offboardVelocityModeSelected(_offboard_control_mode);

	if (mode == ActiveMode::Offboard) {
		if (_active_reference_source == ReferenceSource::Trajectory && !offboard_velocity_mode) {
			if (!_pose_trajectory.finished(now)) {
				return;
			}

			if (!_pose_trajectory.waypointHoldsAtEnd()) {
				_pose_trajectory.cancelWaypoints(now);
			}

			_active_reference_source = ReferenceSource::Hold;
			_active_trajectory_id = 0;
			_active_trajectory_point_count = 0;
			_active_trajectory_min_z_ned = NAN;
			_last_pose_target_valid = false;
		}
	}

	if (_last_pose_reference_update != 0
	    && now < _last_pose_reference_update + kPoseReferenceUpdateInterval) {
		return;
	}

	_last_pose_reference_update = now;

	if (mode == ActiveMode::Manual) {
		matrix::Vector3f velocity_ned{};

		for (int axis = 0; axis < 3; ++axis) {
			velocity_ned(axis) = PX4_ISFINITE(_trajectory_setpoint.velocity[axis])
					     ? _trajectory_setpoint.velocity[axis] : 0.f;
		}

		matrix::Vector3f velocity_w = positionNedToEnu(velocity_ned);
		float yaw_rate_w = PX4_ISFINITE(_trajectory_setpoint.yawspeed)
				   ? -_trajectory_setpoint.yawspeed : 0.f;
		const bool manual_command_active = velocity_w.norm() > kCmdZeroEps || fabsf(yaw_rate_w) > kCmdZeroEps;
		const AmPoseTrajectory::VelocityConstraints manual_constraints {
			_param_ampc_man_acc_hor.get(),
			_param_ampc_man_jerk_hor.get(),
			_param_ampc_man_acc_up.get(),
			_param_ampc_man_acc_dn.get(),
			_param_ampc_man_jerk_z.get(),
			_param_ampc_man_y_acc.get(),
			_param_ampc_man_y_jerk.get(),
		};

		if (manual_command_active) {
			_manual_hold_anchor_active = false;
			_last_pose_target_valid = false;
			_pose_trajectory.updateVelocity(velocity_w, yaw_rate_w, now, manual_constraints,
							AmPoseTrajectory::VelocitySource::Manual);
			return;
		}

		velocity_w.zero();
		yaw_rate_w = 0.f;

		if (!_manual_hold_anchor_active) {
			_pose_trajectory.updateVelocity(velocity_w, yaw_rate_w, now, manual_constraints,
							AmPoseTrajectory::VelocitySource::Manual);

			if (!_pose_trajectory.velocityAtRest()) {
				_last_pose_target_valid = false;
				return;
			}
		}

		const AmPoseTrajectory::Sample current = _pose_trajectory.sample(now, 0.f);
		matrix::Vector3f target_ned = positionEnuToNed(current.position);

		for (int axis = 0; axis < 3; ++axis) {
			if (PX4_ISFINITE(_trajectory_setpoint.position[axis])) {
				target_ned(axis) = _trajectory_setpoint.position[axis];
			}
		}

		const float target_yaw = PX4_ISFINITE(_trajectory_setpoint.yaw)
					 ? yawNedToEnu(_trajectory_setpoint.yaw) : current.yaw;
		const matrix::Vector3f target_position = positionNedToEnu(target_ned);
		const bool target_changed = !_last_pose_target_valid
					    || (target_position - _last_pose_target_position).norm() > kPoseTargetPositionEpsilon
					    || fabsf(matrix::wrap_pi(target_yaw - _last_pose_target_yaw)) > kPoseTargetYawEpsilon;

		if (target_changed) {
			// FlightTaskManualAmPose exposes this lock only after AMPC_HOLD_MAX_*
			// allows it. The manual smoother has already reached rest, so this
			// one-time quintic anchors the reference to that hold point C2-continuously.
			_pose_trajectory.replanPosition(target_position, target_yaw, now,
							math::max(_param_ampc_vel_manual.get(), 0.1f),
							math::radians(math::max(_param_ampc_man_y_max.get(), 1.f)));
			_last_pose_target_position = target_position;
			_last_pose_target_yaw = target_yaw;
			_last_pose_target_valid = true;
		}

		_manual_hold_anchor_active = true;
		_offboard_velocity_setpoint_lost = false;

		return;
	}

	if (mode != ActiveMode::Offboard || !offboardControlModeSupported()) {
		return;
	}

	const float maximum_velocity = math::constrain(_param_ampc_pose_vmax.get(), 0.1f, 1.f);
	const float maximum_yaw_rate = math::radians(math::constrain(_param_ampc_pose_ymax.get(), 1.f, 57.3f));
	const AmPoseTrajectory::VelocityConstraints velocity_constraints {
		_param_ampc_man_acc_hor.get(),
		_param_ampc_man_jerk_hor.get(),
		_param_ampc_man_acc_up.get(),
		_param_ampc_man_acc_dn.get(),
		_param_ampc_man_jerk_z.get(),
		_param_ampc_man_y_acc.get(),
		_param_ampc_man_y_jerk.get(),
	};

	if (offboard_velocity_mode) {
		const bool setpoint_lost = trajectorySetpointLost();
		const bool setpoint_fresh = trajectorySetpointFresh() && !setpoint_lost;
		const bool new_setpoint = setpoint_fresh && _trajectory_setpoint.timestamp != 0
					  && _trajectory_setpoint.timestamp != _last_velocity_setpoint_timestamp;

		if (new_setpoint) {
			_last_velocity_setpoint_timestamp = _trajectory_setpoint.timestamp;
			_offboard_velocity_hold_active = false;
		}

		_offboard_velocity_setpoint_lost = setpoint_lost;

		if (!setpoint_fresh && _offboard_velocity_hold_active) {
			return;
		}

		matrix::Vector3f velocity_w{};
		float yaw_rate_w{0.f};

		if (setpoint_fresh) {
			matrix::Vector3f velocity_ned{};

			for (int axis = 0; axis < 3; ++axis) {
				velocity_ned(axis) = PX4_ISFINITE(_trajectory_setpoint.velocity[axis])
						     ? _trajectory_setpoint.velocity[axis] : 0.f;
			}

			velocity_w = constrainVelocityCommand(positionNedToEnu(velocity_ned), maximum_velocity);

			yaw_rate_w = PX4_ISFINITE(_trajectory_setpoint.yawspeed)
				     ? math::constrain(-_trajectory_setpoint.yawspeed,
						       -maximum_yaw_rate, maximum_yaw_rate) : 0.f;
		}

		_pose_trajectory.updateVelocity(velocity_w, yaw_rate_w, now, velocity_constraints,
						AmPoseTrajectory::VelocitySource::Offboard);
		_active_reference_source = ReferenceSource::Velocity;
		_active_trajectory_id = 0;
		_active_trajectory_point_count = 0;
		_active_trajectory_min_z_ned = NAN;
		_trajectory_message_valid = false;
		_last_pose_target_valid = false;

		if (!setpoint_fresh && _pose_trajectory.velocityAtRest()) {
			const AmPoseTrajectory::Sample stopped = _pose_trajectory.sample(now, 0.f);
			_pose_trajectory.replanPosition(stopped.position, stopped.yaw, now,
							maximum_velocity, maximum_yaw_rate);
			_active_reference_source = ReferenceSource::Hold;
			_offboard_velocity_hold_active = true;
		}

		return;
	}

	_offboard_velocity_hold_active = false;
	_offboard_velocity_setpoint_lost = false;

	if (!offboardPositionModeSelected(_offboard_control_mode)) {
		return;
	}

	const AmPoseTrajectory::Sample current = _pose_trajectory.sample(now, 0.f);

	if (!trajectorySetpointFresh() || trajectorySetpointLost()) {
		if (_active_reference_source == ReferenceSource::Velocity) {
			_pose_trajectory.replanPosition(current.position, current.yaw, now,
							maximum_velocity, maximum_yaw_rate);
			_active_reference_source = ReferenceSource::Hold;
		}

		return;
	}

	if (_trajectory_setpoint.timestamp == 0
	    || (_trajectory_setpoint.timestamp == _last_point_setpoint_timestamp
		&& _active_reference_source != ReferenceSource::Velocity)) {
		return;
	}

	_last_point_setpoint_timestamp = _trajectory_setpoint.timestamp;

	if (!trajectorySetpointHasPosition(_trajectory_setpoint)) {
		if (_active_reference_source == ReferenceSource::Velocity) {
			_pose_trajectory.replanPosition(current.position, current.yaw, now,
							maximum_velocity, maximum_yaw_rate);
			_active_reference_source = ReferenceSource::Hold;
		}

		return;
	}

	matrix::Vector3f target_ned = positionEnuToNed(current.position);

	for (int axis = 0; axis < 3; ++axis) {
		if (PX4_ISFINITE(_trajectory_setpoint.position[axis])) {
			target_ned(axis) = _trajectory_setpoint.position[axis];
		}
	}

	const matrix::Vector3f target_position = positionNedToEnu(target_ned);
	const float target_yaw = PX4_ISFINITE(_trajectory_setpoint.yaw)
				 ? yawNedToEnu(_trajectory_setpoint.yaw) : current.yaw;
	const bool unfinished_target = (target_position - current.position).norm() > kPoseTargetPositionEpsilon
				       || fabsf(matrix::wrap_pi(target_yaw - current.yaw)) > kPoseTargetYawEpsilon;
	const bool target_changed = !_last_pose_target_valid
				    || (target_position - _last_pose_target_position).norm() > kPoseTargetPositionEpsilon
				    || fabsf(matrix::wrap_pi(target_yaw - _last_pose_target_yaw)) > kPoseTargetYawEpsilon
				    || (_pose_trajectory.finished(now) && unfinished_target);

	if (target_changed) {
		_pose_trajectory.replanPosition(target_position, target_yaw, now, maximum_velocity, maximum_yaw_rate);
		_last_pose_target_position = target_position;
		_last_pose_target_yaw = target_yaw;
		_last_pose_target_valid = true;
		_active_reference_source = ReferenceSource::Point;
	}
}

void AmPoseControl::buildObservation(AmPosePolicyAdapter::Observation &observation, hrt_abstime now)
{
	PoseObservationState state{};
	state.position_w = _root_pos_w;
	state.attitude_w = _root_quat_w;
	state.linear_velocity_b = _root_lin_vel_b;
	state.angular_velocity_b = _root_ang_vel_b;

	for (int joint = 0; joint < kPolicyArmJointDim; ++joint) {
		state.arm_position[joint] = _policy_arm_position[joint];
		state.arm_velocity[joint] = _policy_arm_velocity[joint];
	}

	for (int motor = 0; motor < kActionDim; ++motor) {
		state.last_action[motor] = _prev_action[motor];
	}

	fillPolicyInput(observation, _pose_trajectory, state, now, _adapter.observationContract());
}

void AmPoseControl::beginPolicyControl(ActiveMode mode, hrt_abstime now)
{
	if (mode == _policy_mode) {
		return;
	}

	for (int motor = 0; motor < kActionDim; ++motor) {
		_blend_start_action[motor] = _prev_action[motor];
	}

	_adapter.reset();

	if (mode != ActiveMode::None
	    && !(mode == ActiveMode::Offboard && _active_reference_source == ReferenceSource::Trajectory)) {
		resetPoseReference(now);
	}

	_policy_mode = mode;
	_blend_start = now;
}

void AmPoseControl::blendPolicyAction(const float requested[kActionDim], float blended[kActionDim], hrt_abstime now) const
{
	const float duration_s = math::max(_param_ampc_sw_blend_t.get(), 0.f);
	const float alpha = duration_s <= FLT_EPSILON || _blend_start == 0
			    ? 1.f : math::constrain((now - _blend_start) * 1e-6f / duration_s, 0.f, 1.f);
	blendPolicyActions(_blend_start_action, requested, alpha, blended);
}

void AmPoseControl::updateActionHistory(const AmPosePolicyAdapter::Action &action)
{
	for (int i = 0; i < kActionDim; ++i) {
		_prev_action[i] = action[i];
	}
}

void AmPoseControl::resetActionHistory()
{
	for (int i = 0; i < kActionDim; ++i) {
		_prev_action[i] = 0.0f;
	}
}

bool AmPoseControl::handoverActionFresh(hrt_abstime now) const
{
	return _handover_action_valid && handoverActionFreshAt(_handover_action_timestamp, now);
}

void AmPoseControl::publishPolicyObservation(const AmPosePolicyAdapter::Observation &observation,
		const AmPosePolicyAdapter::Action &action, const actuator_motors_s &actuator_motors,
		uint32_t degraded_flags, const PolicyObservationTiming &timing)
{
	am_pose_policy_observation_s policy_observation{};
	fillPolicyObservation(policy_observation, observation, action, actuator_motors, degraded_flags, timing,
			      _adapter.observationContract());
	_policy_observation_pub.publish(policy_observation);
}

void AmPoseControl::applyAction(const AmPosePolicyAdapter::Observation &observation,
				const AmPosePolicyAdapter::Action &action, AmPosePolicyAdapter::Action &executed_action,
				uint32_t degraded_flags, const PolicyObservationTiming &timing)
{
	actuator_motors_s actuator_motors{};
	const hrt_abstime now = hrt_absolute_time();
	AmPosePolicyAdapter::Action blended_action{};
	blendPolicyAction(action, blended_action, now);
	fillMotorSetpointFromAction(actuator_motors, executed_action, blended_action, now, _angular_velocity.timestamp_sample);

	publishPolicyObservation(observation, action, actuator_motors, degraded_flags, timing);
	_actuator_motors_pub.publish(actuator_motors);

	vehicle_thrust_setpoint_s thrust_setpoint{};
	fillThrustSetpointFromMotors(thrust_setpoint, actuator_motors.timestamp, actuator_motors.timestamp_sample,
				     actuator_motors);
	_vehicle_thrust_setpoint_pub.publish(thrust_setpoint);
}

void AmPoseControl::publishStopSetpoint()
{
	_takeoff_output_released = false;
	_handover_action_consumed = false;
	_handover_entry_accepted = false;

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

void AmPoseControl::publishIdleSetpoint()
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

void AmPoseControl::Run()
{
	if (should_exit()) {
		publishStopSetpoint();
		_angular_velocity_sub.unregisterCallback();
		ScheduleClear();
		exit_and_cleanup();
		return;
	}

	perf_begin(_loop_perf);
	vehicle_angular_velocity_s angular_velocity{};

	if (!_angular_velocity_sub.update(&angular_velocity)) {
		if (_use_am_mode && !angularVelocityValid()) {
			_adapter.reset();
			resetActionHistory();
			_policy_mode = ActiveMode::None;
			_blend_start = 0;
			publishStopSetpoint();
		}

		perf_end(_loop_perf);
		return;
	}

	_angular_velocity = angular_velocity;

	if (!policyStepDue(_angular_velocity.timestamp_sample, _next_policy_deadline)) {
		perf_end(_loop_perf);
		return;
	}

	const bool was_using_am_mode = _use_am_mode;

	if (_parameter_update_sub.updated()) {
		parameter_update_s param_update{};
		_parameter_update_sub.copy(&param_update);
		updateParams();
		_takeoff.setSpoolupTime(_param_com_spoolup_time.get());
		_takeoff.setTakeoffRampTime(_param_ampc_tko_ramp_t.get());
	}

	float dt_s = 0.f;
	const bool vehicle_state_updated = updateVehicleState(dt_s);
	_vehicle_control_mode_sub.update(&_vehicle_control_mode);
	_vehicle_land_detected_sub.update(&_vehicle_land_detected);
	vehicle_status_s vehicle_status{};
	_vehicle_status_sub.copy(&vehicle_status);
	_use_am_mode = vehicle_status.nav_state == vehicle_status_s::NAVIGATION_STATE_AM_POSE
		       || (vehicle_status.nav_state == vehicle_status_s::NAVIGATION_STATE_OFFBOARD
			   && mode_util::isAmPoseOffboardControlMode(_vehicle_control_mode));
	const ActiveMode mode = activeMode();

	if (!_use_am_mode && was_using_am_mode) {
		_handover_action_consumed = false;
		_handover_entry_accepted = false;
	}

	if (!_use_am_mode && !was_using_am_mode && _vehicle_control_mode.flag_control_allocation_enabled) {
		actuator_motors_s allocator_motors{};

		if (_actuator_motors_sub.update(&allocator_motors)) {
			float candidate[kActionDim] {};

			if (allocator_motors.timestamp != 0 && extractFiniteMotorAction(allocator_motors, candidate)) {
				for (int motor = 0; motor < kActionDim; ++motor) {
					_handover_action[motor] = candidate[motor];
				}

				_handover_action_timestamp = allocator_motors.timestamp;
				_handover_action_valid = true;
			}
		}
	}

	publishStatus();

	if (vehicle_state_updated) {
		const bool position_reset = _estimator_reset_counters_initialized
					    && (_position.xy_reset_counter != _xy_reset_counter
						|| _position.z_reset_counter != _z_reset_counter);
		const bool heading_reset = _estimator_reset_counters_initialized
					   && _position.heading_reset_counter != _heading_reset_counter;
		const bool attitude_reset = _estimator_reset_counters_initialized
					    && _attitude.quat_reset_counter != _attitude_reset_counter;

		if (_use_am_mode && (position_reset || heading_reset || attitude_reset)) {
			matrix::Vector3f position_delta_w{};

			if (_position.xy_reset_counter != _xy_reset_counter
			    && PX4_ISFINITE(_position.delta_xy[0]) && PX4_ISFINITE(_position.delta_xy[1])) {
				position_delta_w = positionNedToEnu({_position.delta_xy[0], _position.delta_xy[1], 0.f});
			}

			if (_position.z_reset_counter != _z_reset_counter && PX4_ISFINITE(_position.delta_z)) {
				position_delta_w(2) = -_position.delta_z;

				if (PX4_ISFINITE(_active_trajectory_min_z_ned)) {
					_active_trajectory_min_z_ned += _position.delta_z;
				}
			}

			const float yaw_delta_w = heading_reset && PX4_ISFINITE(_position.delta_heading)
						  ? -_position.delta_heading : 0.f;
			_pose_trajectory.shiftFrame(position_delta_w, yaw_delta_w);

			if (_last_pose_target_valid) {
				_last_pose_target_position += position_delta_w;
				_last_pose_target_yaw = matrix::wrap_pi(_last_pose_target_yaw + yaw_delta_w);
			}

			for (int motor = 0; motor < kActionDim; ++motor) {
				_blend_start_action[motor] = _prev_action[motor];
			}

			_blend_start = hrt_absolute_time();
			_adapter.reset();
		}

		_xy_reset_counter = _position.xy_reset_counter;
		_z_reset_counter = _position.z_reset_counter;
		_heading_reset_counter = _position.heading_reset_counter;
		_attitude_reset_counter = _attitude.quat_reset_counter;
		_estimator_reset_counters_initialized = true;
	}

	if (_use_am_mode && !was_using_am_mode) {
		const hrt_abstime now = hrt_absolute_time();
		const bool airborne = _vehicle_control_mode.flag_armed && !_vehicle_land_detected.landed;

		if (airborne && _handover_action_valid) {
			for (int motor = 0; motor < kActionDim; ++motor) {
				_prev_action[motor] = _handover_action[motor];
			}

			_handover_action_consumed = handoverActionFresh(now);
			_handover_entry_accepted = _handover_action_consumed;

			if (!_handover_action_consumed) {
				PX4_WARN("AM Pose entered with stale allocator handover");
			}

		} else {
			_handover_action_consumed = false;
			_handover_entry_accepted = !airborne;
		}

		_takeoff_output_released = false;
		_policy_mode = ActiveMode::None;
		_blend_start = 0;
		_adapter.reset();

		if (vehicle_state_updated) {
			resetPoseReference(hrt_absolute_time());
		}
	}

	if (!_use_am_mode) {
		if (was_using_am_mode) {
			publishStopSetpoint();
			resetActionHistory();
			_handover_action_valid = false;
			_handover_action_timestamp = 0;
			_policy_mode = ActiveMode::None;
			_blend_start = 0;
			_adapter.reset();
			_trajectory_message_valid = false;
			_active_reference_source = ReferenceSource::Hold;
			_active_trajectory_id = 0;
			_last_accepted_trajectory_id = 0;
			_has_accepted_trajectory_id = false;
			_active_trajectory_point_count = 0;
			_active_trajectory_min_z_ned = NAN;
			_offboard_velocity_hold_active = false;
			_offboard_velocity_setpoint_lost = false;
		}

		perf_end(_loop_perf);
		return;
	}

	uint32_t degraded_flags = am_pose_policy_observation_s::DEGRADED_NONE;

	if (!vehicle_state_updated || !armStateValid()) {
		_adapter.reset();
		resetActionHistory();
		_policy_mode = ActiveMode::None;
		_blend_start = 0;
		publishStopSetpoint();
		perf_end(_loop_perf);
		return;
	}

	updatePolicyArmJointState();
	_trajectory_setpoint_sub.update(&_trajectory_setpoint);
	_am_pose_trajectory_sub.update(&_am_pose_trajectory);
	updateOffboardControlMode();

	const bool trajectory_setpoint_fresh = trajectorySetpointFresh();
	const bool trajectory_setpoint_lost = trajectorySetpointLost();

	if (!trajectory_setpoint_fresh && !trajectory_setpoint_lost) {
		degraded_flags |= kDegradedSetpointHeld;
	}

	bool trajectory_setpoint_valid = !trajectory_setpoint_lost && trajectorySetpointHasLinearInput();

	if (mode == ActiveMode::Offboard) {
		const bool external_setpoint_valid = offboardTrajectorySetpointValid();
		const bool unsupported_external_input = offboardAmPoseInputRejected(_offboard_control_mode, _trajectory_setpoint,
							offboardControlModeFresh(), trajectory_setpoint_fresh);

		// No command is a valid AM Pose Offboard state: beginPolicyControl() resets
		// the trajectory to the measured pose and the policy holds it. Fresh
		// position commands plan reach-and-hold, while velocity commands drive a
		// jerk-limited pose reference.
		trajectory_setpoint_valid = !unsupported_external_input;

		if (!external_setpoint_valid) {
			degraded_flags |= kDegradedSetpointHeld;
		}
	}

	if (!trajectory_setpoint_valid) {
		if (hrt_elapsed_time(&_last_setpoint_diag) > 1_s) {
			_last_setpoint_diag = hrt_absolute_time();
			PX4_WARN("AM Pose waiting for a valid trajectory_setpoint");
		}

		_adapter.reset();
		resetActionHistory();
		_policy_mode = ActiveMode::None;
		_blend_start = 0;
		publishStopSetpoint();
		perf_end(_loop_perf);
		return;
	}

	if (mode == ActiveMode::Offboard && offboardControlModeValid()
	    && offboardPositionModeSelected(_offboard_control_mode)) {
		const hrt_abstime trajectory_now = hrt_absolute_time();

		if (!_pose_trajectory.initialized()) {
			resetPoseReference(trajectory_now);
		}

		if (_am_pose_trajectory.timestamp != 0
		    && _am_pose_trajectory.timestamp != _last_trajectory_message_timestamp) {
			perf_begin(_trajectory_command_perf);
			processTrajectoryCommand(trajectory_now,
						 math::constrain(_param_ampc_pose_vmax.get(), 0.1f, 1.f),
						 math::radians(math::constrain(_param_ampc_pose_ymax.get(), 1.f, 57.3f)));
			perf_end(_trajectory_command_perf);
		}
	}

	if (updateTakeoffGate(mode, was_using_am_mode, dt_s)) {
		_adapter.reset();
		resetActionHistory();
		_policy_mode = ActiveMode::None;
		_blend_start = 0;
		publishIdleSetpoint();
		perf_end(_loop_perf);
		return;
	}

	if (!_policy_loaded) {
		if (hrt_elapsed_time(&_last_setpoint_diag) > 1_s) {
			_last_setpoint_diag = hrt_absolute_time();
			PX4_WARN("AM Pose policy unavailable");
		}

		_adapter.reset();
		resetActionHistory();
		_policy_mode = ActiveMode::None;
		_blend_start = 0;
		publishStopSetpoint();
		perf_end(_loop_perf);
		return;
	}

	const hrt_abstime now = hrt_absolute_time();
	beginPolicyControl(mode, now);
	perf_begin(_trajectory_reference_perf);
	updatePoseReference(mode, now);
	perf_end(_trajectory_reference_perf);

	const bool commit_policy_state = _takeoff_output_released
					 || takeoffStateAllowsPolicyStateCommit(static_cast<uint8_t>(_takeoff.getTakeoffState()));

	if (!commit_policy_state) {
		_adapter.reset();
		resetActionHistory();
	}

	PolicyObservationTiming policy_timing{};
	policy_timing.observation_build_timestamp = now;
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
	policy_timing.am_pose_trajectory_timestamp = _am_pose_trajectory.timestamp;
	policy_timing.offboard_control_mode_timestamp = _offboard_control_mode.timestamp;
	policy_timing.takeoff_state = static_cast<uint8_t>(_takeoff.getTakeoffState());

	AmPosePolicyAdapter::Observation observation{};
	buildObservation(observation, now);
	AmPosePolicyAdapter::Action action{};
	policy_timing.policy_inference_start_timestamp = hrt_absolute_time();
	bool inference_ok = _adapter.infer(observation, action);
	inference_ok = inference_ok && actionFinite(action);
	policy_timing.policy_inference_finish_timestamp = hrt_absolute_time();

	if (inference_ok) {
		AmPosePolicyAdapter::Action executed_action{};
		policy_timing.policy_sequence = ++_policy_sequence;
		applyAction(observation, action, executed_action, degraded_flags, policy_timing);

		if (commit_policy_state) {
			updateActionHistory(executed_action);
		}

	} else {
		_adapter.reset();
		publishStopSetpoint();
		resetActionHistory();
		_policy_mode = ActiveMode::None;
		_blend_start = 0;
	}

	perf_end(_loop_perf);
}

int AmPoseControl::task_spawn(int argc, char *argv[])
{
	AmPoseControl *instance = new AmPoseControl();

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

int AmPoseControl::custom_command(int argc, char *argv[])
{
	return print_usage("unknown command");
}

int AmPoseControl::print_status()
{
	updateOffboardControlMode();
	_trajectory_setpoint_sub.update(&_trajectory_setpoint);
	const ActiveMode mode = activeMode();
	const bool manual_control_available = manualControlAvailable();
	const bool vehicle_state_valid = vehicleStateValid();
	const bool attitude_valid = attitudeValid();
	const bool angular_velocity_valid = angularVelocityValid();
	const bool arm_state_valid = armStateValid();
	const bool trajectory_setpoint_valid = trajectorySetpointValid(mode);
	const bool am_pose_available = manual_control_available && arm_state_valid && _policy_loaded;
	const bool handover_required = _vehicle_control_mode.flag_armed && !_vehicle_land_detected.landed;
	const bool handover_ready = handoverActionReady(handover_required, _handover_entry_accepted,
				    handoverActionFresh(hrt_absolute_time()));
	const bool offboard_am_pose_available = arm_state_valid && _policy_loaded && handover_ready;
	const bool offboard_control_mode_fresh = offboardControlModeFresh();
	const bool offboard_control_mode_supported = offboardControlModeSupported();
	const bool offboard_control_mode_valid = offboardControlModeValid();

	const bool manual_control_required = mode != ActiveMode::Offboard;
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
	PX4_INFO("policy_loaded: %s, am_pose_available: %s, offboard_am_pose_available: %s",
		 _policy_loaded ? "yes" : "no",
		 (am_pose_available && handover_ready) ? "yes" : "no",
		 offboard_am_pose_available ? "yes" : "no");
	PX4_INFO("handover_action_ready: %s, consumed: %s, age: %.3f s",
		 handover_ready ? "yes" : "no",
		 _handover_action_consumed ? "yes" : "no",
		 _handover_action_timestamp != 0 && hrt_absolute_time() >= _handover_action_timestamp
		 ? (hrt_absolute_time() - _handover_action_timestamp) * 1e-6 : static_cast<double>(NAN));
	perf_print_counter(_trajectory_reference_perf);
	perf_print_counter(_trajectory_command_perf);
	perf_print_counter(_loop_perf);

	return 0;
}

int AmPoseControl::print_usage(const char *reason)
{
	if (reason) {
		PX4_ERR("%s", reason);
	}

	PRINT_MODULE_DESCRIPTION(
		R"DESCR_STR(
### Description
AM Pose direct-actuator controller for manual AM Pose and standard Offboard.

The controller tracks a C2 position and yaw-only preview with the 134D AM Pose
policy. AM Pose Offboard accepts position reach-and-hold or jerk-limited world-frame
velocity and yaw-rate commands. Position has priority when both modes are set.
)DESCR_STR");

	PRINT_MODULE_USAGE_NAME("am_pose_control", "controller");
	PRINT_MODULE_USAGE_COMMAND("start");
	PRINT_MODULE_USAGE_DEFAULT_COMMANDS();
	return 0;
}

extern "C" __EXPORT int am_pose_control_main(int argc, char *argv[])
{
	return AmPoseControl::main(argc, argv);
}
