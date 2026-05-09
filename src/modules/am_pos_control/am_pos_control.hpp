/****************************************************************************
 *
 * AM Position direct actuator controller.
 *
 ****************************************************************************/
#pragma once

#include <lib/sticks/Sticks.hpp>
#include <mathlib/math/Functions.hpp>
#include <mathlib/mathlib.h>
#include <matrix/matrix/math.hpp>
#include <perf/perf_counter.h>
#include <px4_platform_common/defines.h>
#include <px4_platform_common/log.h>
#include <px4_platform_common/module.h>
#include <px4_platform_common/module_params.h>
#include <px4_platform_common/px4_config.h>
#include <px4_platform_common/px4_work_queue/WorkItem.hpp>
#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/SubscriptionCallback.hpp>
#include <uORB/topics/actuator_motors.h>
#include <uORB/topics/am_pos_control_status.h>
#include <uORB/topics/am_policy_observation.h>
#include <uORB/topics/am_test_result.h>
#include <uORB/topics/am_test_status.h>
#include <uORB/topics/arm_joint_state.h>
#include <uORB/topics/offboard_control_mode.h>
#include <uORB/topics/parameter_update.h>
#include <uORB/topics/trajectory_setpoint.h>
#include <uORB/topics/vehicle_constraints.h>
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/vehicle_angular_velocity.h>
#include <uORB/topics/vehicle_control_mode.h>
#include <uORB/topics/vehicle_land_detected.h>
#include <uORB/topics/vehicle_local_position.h>
#include <uORB/topics/vehicle_status.h>
#include <uORB/topics/vehicle_thrust_setpoint.h>

#include <Takeoff.hpp>

#include "rl_tools_adapter.hpp"

using namespace time_literals;

class AmPosControl : public ModuleBase<AmPosControl>, public ModuleParams, public px4::WorkItem
{
public:
	AmPosControl();
	~AmPosControl() override;

	static int task_spawn(int argc, char *argv[]);
	static int custom_command(int argc, char *argv[]);
	static int print_usage(const char *reason = nullptr);
	int print_status() override;

	bool init();
	static void fillAmTestResultFromAction(am_test_result_s &result, hrt_abstime now, hrt_abstime timestamp_sample,
					       hrt_abstime am_setpoint_timestamp, const RlToolsAdapter::Action &action,
					       uint32_t degraded_flags = am_test_result_s::DEGRADED_NONE)
	{
		result = {};
		result.timestamp = now;
		result.timestamp_sample = timestamp_sample;
		result.am_valid = true;
		result.failure_flags = am_test_result_s::FAILURE_NONE;
		result.degraded_flags = degraded_flags;
		result.am_setpoint_timestamp = am_setpoint_timestamp;

		for (int i = 0; i < kActionDim; ++i) {
			const float mapped = math::constrain(1.0f / (1.0f + expf(-2.0f * action[i])), 0.0f, 1.0f);
			result.am_raw_action[i] = action[i];
			result.am_mapped_action[i] = mapped;
			result.am_motor_control[i] = mapped;
		}

		for (int i = kActionDim; i < kMotorControlDim; ++i) {
			result.am_motor_control[i] = NAN;
		}
	}

	static void fillInvalidAmTestResult(am_test_result_s &result, hrt_abstime now, hrt_abstime timestamp_sample,
					    hrt_abstime am_setpoint_timestamp, uint32_t failure_flags,
					    uint32_t degraded_flags = am_test_result_s::DEGRADED_NONE)
	{
		result = {};
		result.timestamp = now;
		result.timestamp_sample = timestamp_sample;
		result.am_valid = false;
		result.failure_flags = failure_flags;
		result.degraded_flags = degraded_flags;
		result.am_setpoint_timestamp = am_setpoint_timestamp;

		for (float &value : result.am_raw_action) {
			value = NAN;
		}

		for (float &value : result.am_mapped_action) {
			value = NAN;
		}

		for (float &value : result.am_motor_control) {
			value = NAN;
		}
	}

	static bool vehicleStateValidStrict(const vehicle_local_position_s &position, bool attitude_valid,
					    bool angular_velocity_valid, hrt_abstime now)
	{
		if ((position.timestamp == 0) || (now > position.timestamp + kStateTimeout)) {
			return false;
		}

		const bool xy_valid = position.xy_valid && PX4_ISFINITE(position.x) && PX4_ISFINITE(position.y);
		const bool z_valid = position.z_valid && PX4_ISFINITE(position.z);
		const bool v_xy_valid = position.v_xy_valid && PX4_ISFINITE(position.vx) && PX4_ISFINITE(position.vy);
		const bool v_z_valid = position.v_z_valid && PX4_ISFINITE(position.vz);
		const bool heading_valid = PX4_ISFINITE(position.heading);

		return xy_valid && z_valid && v_xy_valid && v_z_valid && heading_valid && attitude_valid && angular_velocity_valid;
	}

	static bool vehicleStateValidForAmTest(const vehicle_local_position_s &position, bool attitude_valid,
					       bool angular_velocity_valid, hrt_abstime now, uint32_t &degraded_flags)
	{
		degraded_flags = am_test_result_s::DEGRADED_NONE;

		if ((position.timestamp == 0) || (now > position.timestamp + kStateTimeout)) {
			return false;
		}

		if (!position.xy_valid) {
			degraded_flags |= am_test_result_s::DEGRADED_LOCAL_XY_INVALID;
		}

		if (!position.v_xy_valid) {
			degraded_flags |= am_test_result_s::DEGRADED_LOCAL_VXY_INVALID;
		}

		const bool xy_values_finite = PX4_ISFINITE(position.x) && PX4_ISFINITE(position.y);
		const bool z_valid = position.z_valid && PX4_ISFINITE(position.z);
		const bool v_xy_values_finite = PX4_ISFINITE(position.vx) && PX4_ISFINITE(position.vy);
		const bool v_z_valid = position.v_z_valid && PX4_ISFINITE(position.vz);
		const bool heading_valid = PX4_ISFINITE(position.heading);

		return xy_values_finite && z_valid && v_xy_values_finite && v_z_valid && heading_valid
		       && attitude_valid && angular_velocity_valid;
	}

	static void fillDefaultAmTestSetpoint(trajectory_setpoint_s &setpoint, hrt_abstime now,
					      const vehicle_local_position_s &position, uint32_t &degraded_flags)
	{
		setpoint = {};
		setpoint.timestamp = now;
		setpoint.position[0] = position.x;
		setpoint.position[1] = position.y;
		setpoint.position[2] = position.z;
		setpoint.velocity[0] = 0.0f;
		setpoint.velocity[1] = 0.0f;
		setpoint.velocity[2] = 0.0f;
		setpoint.yaw = position.heading;
		setpoint.yawspeed = 0.0f;
		degraded_flags |= am_test_result_s::DEGRADED_SETPOINT_DEFAULTED;
	}

	static void fillThrustSetpointFromMotors(vehicle_thrust_setpoint_s &thrust_setpoint, hrt_abstime now,
			hrt_abstime timestamp_sample, const actuator_motors_s &actuator_motors)
	{
		float thrust_sum = 0.0f;
		int thrust_count = 0;

		for (int i = 0; i < kActionDim; ++i) {
			if (PX4_ISFINITE(actuator_motors.control[i])) {
				thrust_sum += actuator_motors.control[i];
				++thrust_count;
			}
		}

		thrust_setpoint = {};
		thrust_setpoint.timestamp = now;
		thrust_setpoint.timestamp_sample = timestamp_sample;
		thrust_setpoint.xyz[0] = 0.0f;
		thrust_setpoint.xyz[1] = 0.0f;
		thrust_setpoint.xyz[2] = thrust_count > 0 ? -(thrust_sum / static_cast<float>(thrust_count)) : 0.0f;
	}

	static void fillIdleMotorSetpoint(actuator_motors_s &actuator_motors, hrt_abstime now, hrt_abstime timestamp_sample)
	{
		actuator_motors = {};
		actuator_motors.timestamp = now;
		actuator_motors.timestamp_sample = timestamp_sample;

		for (int i = 0; i < kActionDim; ++i) {
			actuator_motors.control[i] = 0.0f;
		}

		for (int i = kActionDim; i < kMotorControlDim; ++i) {
			actuator_motors.control[i] = NAN;
		}

		actuator_motors.reversible_flags = 0;
	}

	static float mapActionToMotor(float action)
	{
		return math::constrain(1.0f / (1.0f + expf(-2.0f * action)), 0.0f, 1.0f);
	}

	static float inverseMappedMotorToAction(float motor_control)
	{
		constexpr float kMotorEps = 1e-4f;
		const float motor = math::constrain(motor_control, kMotorEps, 1.0f - kMotorEps);
		return 0.5f * logf(motor / (1.0f - motor));
	}

	static float manualTakeoffReleaseFromThrottle(float throttle_zero_centered, float deadzone)
	{
		deadzone = math::constrain(deadzone, 0.0f, 0.99f);
		const float release = (throttle_zero_centered + 1.0f - deadzone) / (1.0f - deadzone);
		return release <= FLT_EPSILON ? 0.0f : math::constrain(release, 0.0f, 1.0f);
	}

	static void applyMotorRelease(actuator_motors_s &actuator_motors, float release)
	{
		release = math::constrain(release, 0.0f, 1.0f);

		for (int i = 0; i < kActionDim; ++i) {
			const float control = PX4_ISFINITE(actuator_motors.control[i]) ? actuator_motors.control[i] : 0.0f;
			actuator_motors.control[i] = math::constrain(control * release, 0.0f, 1.0f);
		}

		for (int i = kActionDim; i < kMotorControlDim; ++i) {
			actuator_motors.control[i] = NAN;
		}
	}

	static void fillActionHistoryFromMotors(RlToolsAdapter::Action &action_history, const actuator_motors_s &actuator_motors)
	{
		for (int i = 0; i < kActionDim; ++i) {
			const float control = PX4_ISFINITE(actuator_motors.control[i]) ? actuator_motors.control[i] : 0.0f;
			action_history[i] = inverseMappedMotorToAction(control);
		}
	}

	static void rampMotorOutputsForTakeoff(actuator_motors_s &actuator_motors, float ramp_progress)
	{
		ramp_progress = math::constrain(ramp_progress, 0.0f, 1.0f);

		float collective = 0.0f;

		for (int i = 0; i < kActionDim; ++i) {
			collective += PX4_ISFINITE(actuator_motors.control[i]) ? actuator_motors.control[i] : 0.0f;
		}

		collective /= static_cast<float>(kActionDim);
		const float ramped_collective = collective * ramp_progress;

		for (int i = 0; i < kActionDim; ++i) {
			const float control = PX4_ISFINITE(actuator_motors.control[i]) ? actuator_motors.control[i] : collective;
			const float differential = control - collective;
			actuator_motors.control[i] = math::constrain(ramped_collective + differential * ramp_progress,
						       0.0f, 1.0f);
		}

		for (int i = kActionDim; i < kMotorControlDim; ++i) {
			actuator_motors.control[i] = NAN;
		}
	}

	static float advanceAmTakeoffRampProgress(float current_progress, float dt_s, float ramp_time_s,
			uint8_t takeoff_state, bool skip_takeoff)
	{
		if (skip_takeoff || takeoff_state >= takeoff_status_s::TAKEOFF_STATE_FLIGHT) {
			return 1.0f;
		}

		if (takeoff_state < takeoff_status_s::TAKEOFF_STATE_RAMPUP) {
			return 0.0f;
		}

		if (ramp_time_s <= dt_s) {
			return 1.0f;
		}

		return math::constrain(current_progress + dt_s / ramp_time_s, 0.0f, 1.0f);
	}

	static bool offboardSetpointWantsTakeoff(const trajectory_setpoint_s &setpoint,
			const vehicle_local_position_s &position, hrt_abstime timestamp_sample)
	{
		const bool setpoint_recent = timestamp_sample < setpoint.timestamp + 1_s;

		if (setpoint_recent && PX4_ISFINITE(setpoint.position[2]) && PX4_ISFINITE(position.z)
		    && (setpoint.position[2] < position.z)) {
			return true;
		}

		if (setpoint_recent && PX4_ISFINITE(setpoint.velocity[2]) && (setpoint.velocity[2] < 0.f)) {
			return true;
		}

		if (setpoint_recent && PX4_ISFINITE(setpoint.acceleration[2]) && (setpoint.acceleration[2] < 0.f)) {
			return true;
		}

		return false;
	}

	static bool shouldSkipTakeoffRampOnAmModeEntry(bool using_am_mode, bool was_using_am_mode, bool landed)
	{
		return using_am_mode && !was_using_am_mode && !landed;
	}

	static bool takeoffStateRequiresOutputGate(uint8_t takeoff_state, bool ground_contact)
	{
		const bool not_taken_off = takeoff_state < takeoff_status_s::TAKEOFF_STATE_RAMPUP;
		const bool flying_but_ground_contact = (takeoff_state >= takeoff_status_s::TAKEOFF_STATE_FLIGHT) && ground_contact;
		return not_taken_off || flying_but_ground_contact;
	}

	static matrix::Vector3f gatePositionErrorForPolicy(const matrix::Vector3f &pos_error_w,
			const matrix::Quatf &root_quat_w, const matrix::Quatf &heading_quat_w,
			const bool active_lin_h[3])
	{
		matrix::Vector3f hold_pos_error_h = heading_quat_w.inversed().rotateVector(pos_error_w);

		for (int i = 0; i < 3; ++i) {
			if (active_lin_h[i]) {
				hold_pos_error_h(i) = 0.0f;
			}
		}

		return root_quat_w.inversed().rotateVector(heading_quat_w.rotateVector(hold_pos_error_h));
	}

private:
	static constexpr int kActionDim = 4;
	static constexpr int kArmJointDim = 5;
	static constexpr int kMotorControlDim = 12;
	static constexpr float kCmdZeroEps = 1e-6f;
	static constexpr int kStartupDiagSamples = 8;
	static constexpr float kDiagLowMeanCommand = 0.75f;
	static constexpr float kDiagLowMinCommand = 0.45f;
	static constexpr float kDiagWideSpread = 0.35f;
	static constexpr hrt_abstime kStateTimeout = 1_s;
	static constexpr hrt_abstime kArmStateTimeout = 200_ms;
	static constexpr hrt_abstime kTrajectorySetpointTimeout = 200_ms;

	enum class ActiveMode : uint8_t {
		None = 0,
		Manual,
		Offboard,
		Test
	};

	struct CommandReference {
		matrix::Vector3f desired_lin_vel_b{};
		matrix::Vector3f desired_ang_vel_b{};
		matrix::Vector3f desired_pos_w{};
		matrix::Quatf desired_quat_w{};
		bool lin_cmd_active[3]{false, false, false};
		bool ang_cmd_active[3]{false, false, false};
		bool has_lin_vel_cmd{false};
		bool has_ang_vel_cmd{false};
	};

	void Run() override;
	void updateTargets();
	void updateTargets(bool use_default_am_test_setpoint);
	void buildObservation(RlToolsAdapter::Observation &observation);
	void applyAction(const RlToolsAdapter::Observation &observation, const RlToolsAdapter::Action &action,
			 RlToolsAdapter::Action &executed_action, ActiveMode mode, bool publish_outputs,
			 uint32_t degraded_flags = am_policy_observation_s::DEGRADED_NONE);
	void publishPolicyObservation(const RlToolsAdapter::Observation &observation, const RlToolsAdapter::Action &action,
				      const actuator_motors_s &actuator_motors, uint32_t degraded_flags);
	void publishAmTestStatus(bool vehicle_state_valid, bool arm_state_valid, bool am_setpoint_valid, bool am_valid,
				 uint32_t failure_flags, uint32_t degraded_flags);
	void publishAmTestResult(uint32_t failure_flags, uint32_t degraded_flags);
	void publishAmTestResult(const RlToolsAdapter::Action &action, uint32_t degraded_flags);
	void publishStopSetpoint();
	void publishIdleSetpoint();
	void publishTakeoffStatus();
	bool updateTakeoffGate(ActiveMode mode, bool was_using_am_mode, float dt_s);
	void updateActionHistory(const RlToolsAdapter::Action &action);
	void resetActionHistory();
	void maybeLogPolicyDiagnostics(const RlToolsAdapter::Observation &observation, const RlToolsAdapter::Action &action);
	void publishStatus();
	void resetCommandReference();
	void resetState();
	bool updateVehicleState(float &dt_s, bool allow_am_test_degraded, uint32_t &degraded_flags);
	void updateConvertedState();
	bool anyAxisActive(const bool axes[3]) const;
	bool armStateValid() const;
	bool vehicleStateValid() const;
	bool attitudeValid() const;
	bool angularVelocityValid() const;
	bool trajectorySetpointFresh() const;
	bool trajectorySetpointHasLinearInput() const;
	bool offboardTrajectorySetpointValid() const;
	bool trajectorySetpointValid() const;
	bool offboardControlModeFresh() const;
	bool offboardControlModeSupported() const;
	bool offboardControlModeValid() const;
	bool manualControlAvailable();
	ActiveMode activeMode();

	uORB::SubscriptionInterval _parameter_update_sub{ORB_ID(parameter_update), 1_s};
	uORB::Subscription _offboard_control_mode_sub{ORB_ID(offboard_control_mode)};
	uORB::Subscription _vehicle_constraints_sub{ORB_ID(vehicle_constraints)};
	uORB::Subscription _vehicle_control_mode_sub{ORB_ID(vehicle_control_mode)};
	uORB::Subscription _vehicle_land_detected_sub{ORB_ID(vehicle_land_detected)};
	uORB::Subscription _vehicle_status_sub{ORB_ID(vehicle_status)};
	uORB::Subscription _position_sub{ORB_ID(vehicle_local_position)};
	uORB::Subscription _trajectory_setpoint_sub{ORB_ID(trajectory_setpoint)};
	uORB::Subscription _attitude_sub{ORB_ID(vehicle_attitude)};
	uORB::Subscription _arm_joint_state_sub{ORB_ID(arm_joint_state)};
	uORB::SubscriptionCallbackWorkItem _angular_velocity_sub{this, ORB_ID(vehicle_angular_velocity)};

	uORB::Publication<actuator_motors_s> _actuator_motors_pub{ORB_ID(actuator_motors)};
	uORB::Publication<am_pos_control_status_s> _status_pub{ORB_ID(am_pos_control_status)};
	uORB::Publication<am_policy_observation_s> _policy_observation_pub{ORB_ID(am_policy_observation)};
	uORB::Publication<am_test_result_s> _am_test_result_pub{ORB_ID(am_test_result)};
	uORB::Publication<am_test_status_s> _am_test_status_pub{ORB_ID(am_test_status)};
	uORB::PublicationData<takeoff_status_s> _takeoff_status_pub{ORB_ID(takeoff_status)};
	uORB::Publication<vehicle_thrust_setpoint_s> _vehicle_thrust_setpoint_pub{ORB_ID(vehicle_thrust_setpoint)};

	perf_counter_t _loop_perf;
	hrt_abstime _last_run{0};
	hrt_abstime _last_offboard_diag{0};
	hrt_abstime _last_setpoint_diag{0};
	bool _use_am_mode{false};
	CommandReference _current_cmd_ref{};

	offboard_control_mode_s _offboard_control_mode{};
	vehicle_constraints_s _vehicle_constraints{0, NAN, NAN, false, {}};
	vehicle_control_mode_s _vehicle_control_mode{};
	vehicle_land_detected_s _vehicle_land_detected {
		.timestamp = 0,
		.freefall = false,
		.ground_contact = true,
		.maybe_landed = true,
		.landed = true,
	};
	trajectory_setpoint_s _trajectory_setpoint{};
	vehicle_angular_velocity_s _angular_velocity{};
	vehicle_local_position_s _position{};
	vehicle_attitude_s _attitude{};
	arm_joint_state_s _arm_joint_state{};
	matrix::Vector3f _root_pos_w{};
	matrix::Quaternionf _root_quat_w{};
	matrix::Vector3f _root_lin_vel_w{};
	matrix::Vector3f _root_lin_vel_b{};
	matrix::Vector3f _root_ang_vel_b{};
	float _heading_w{0.0f};
	float _prev_action[kActionDim]{0.f, 0.f, 0.f, 0.f};
	float _takeoff_output_ramp_progress{0.0f};
	float _manual_takeoff_release{0.0f};
	int _startup_diag_samples_remaining{0};

	RlToolsAdapter _adapter{};
	TakeoffHandling _takeoff{};
	Sticks _sticks{this};

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::AMPC_VEL_MANUAL>) _param_ampc_vel_manual,
		(ParamFloat<px4::params::AMPC_VEL_SIDE>) _param_ampc_vel_side,
		(ParamFloat<px4::params::AMPC_VEL_BACK>) _param_ampc_vel_back,
		(ParamFloat<px4::params::AMPC_Z_VEL_UP>) _param_ampc_z_vel_up,
		(ParamFloat<px4::params::AMPC_Z_VEL_DN>) _param_ampc_z_vel_dn,
		(ParamFloat<px4::params::AMPC_MAN_Y_MAX>) _param_ampc_man_y_max,
		(ParamFloat<px4::params::AMPC_MAN_Y_TAU>) _param_ampc_man_y_tau,
		(ParamFloat<px4::params::AMPC_MAN_DZ>) _param_ampc_man_dz,
		(ParamFloat<px4::params::AMPC_HOLD_MAX_Z>) _param_ampc_hold_max_z,
		(ParamFloat<px4::params::AMPC_HOLD_MAX_XY>) _param_ampc_hold_max_xy,
		(ParamFloat<px4::params::COM_OF_LOSS_T>) _param_com_of_loss_t,
		(ParamFloat<px4::params::COM_SPOOLUP_TIME>) _param_com_spoolup_time,
		(ParamFloat<px4::params::MPC_TKO_RAMP_T>) _param_mpc_tko_ramp_t
	)
};
