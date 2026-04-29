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
#include <uORB/topics/arm_joint_state.h>
#include <uORB/topics/offboard_control_mode.h>
#include <uORB/topics/parameter_update.h>
#include <uORB/topics/trajectory_setpoint.h>
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/vehicle_angular_velocity.h>
#include <uORB/topics/vehicle_local_position.h>
#include <uORB/topics/vehicle_status.h>

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
		Offboard
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
		void buildObservation(RlToolsAdapter::Observation &observation);
		void applyAction(const RlToolsAdapter::Action &action);
		void publishStopSetpoint();
		void updateActionHistory(const RlToolsAdapter::Action &action);
		void maybeLogPolicyDiagnostics(const RlToolsAdapter::Observation &observation, const RlToolsAdapter::Action &action);
		void publishStatus();
	void resetCommandReference();
	void resetState();
	bool updateVehicleState(float &dt_s);
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
	uORB::Subscription _vehicle_status_sub{ORB_ID(vehicle_status)};
	uORB::Subscription _position_sub{ORB_ID(vehicle_local_position)};
	uORB::Subscription _trajectory_setpoint_sub{ORB_ID(trajectory_setpoint)};
	uORB::Subscription _attitude_sub{ORB_ID(vehicle_attitude)};
	uORB::Subscription _arm_joint_state_sub{ORB_ID(arm_joint_state)};
	uORB::SubscriptionCallbackWorkItem _angular_velocity_sub{this, ORB_ID(vehicle_angular_velocity)};

	uORB::Publication<actuator_motors_s> _actuator_motors_pub{ORB_ID(actuator_motors)};
	uORB::Publication<am_pos_control_status_s> _status_pub{ORB_ID(am_pos_control_status)};

	perf_counter_t _loop_perf;
	hrt_abstime _last_run{0};
	hrt_abstime _last_offboard_diag{0};
	hrt_abstime _last_setpoint_diag{0};
	bool _use_am_mode{false};
	CommandReference _current_cmd_ref{};

	offboard_control_mode_s _offboard_control_mode{};
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
	int _startup_diag_samples_remaining{0};

	RlToolsAdapter _adapter{};
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
		(ParamFloat<px4::params::COM_OF_LOSS_T>) _param_com_of_loss_t
	)
};
