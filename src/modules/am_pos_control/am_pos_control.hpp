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
#include <uORB/topics/arm_joint_state.h>
#include <uORB/topics/arming_check_reply.h>
#include <uORB/topics/arming_check_request.h>
#include <uORB/topics/offboard_control_mode.h>
#include <uORB/topics/parameter_update.h>
#include <uORB/topics/register_ext_component_reply.h>
#include <uORB/topics/register_ext_component_request.h>
#include <uORB/topics/trajectory_setpoint.h>
#include <uORB/topics/unregister_ext_component.h>
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/vehicle_angular_velocity.h>
#include <uORB/topics/vehicle_control_mode.h>
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
	static constexpr hrt_abstime kStateTimeout = 1_s;
	static constexpr hrt_abstime kArmStateTimeout = 200_ms;
	static constexpr hrt_abstime kTrajectorySetpointTimeout = 200_ms;

	enum class ActiveMode : uint8_t {
		None = 0,
		Manual,
		Offboard
	};

	struct RegisteredMode {
		RegisteredMode(const char *mode_name, uint64_t request_id_, bool replace_offboard_)
			: name(mode_name), request_id(request_id_), replace_offboard(replace_offboard_) {}

		const char *name;
		uint64_t request_id;
		bool replace_offboard;
		int8_t arming_check_id{-1};
		int8_t mode_id{-1};
		bool registration_requested{false};

		bool registered() const { return mode_id >= 0 && arming_check_id >= 0; }
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
	void registerMode(RegisteredMode &mode);
	void unregisterMode(RegisteredMode &mode);
	void configureMode(const RegisteredMode &mode);
	void replyToArmingCheck(const RegisteredMode &mode, uint8_t request_id);
	void checkModeRegistration();
	void updateTargets();
	void buildObservation(RlToolsAdapter::Observation &observation);
	void applyAction(const RlToolsAdapter::Action &action);
	void updateActionHistory(const RlToolsAdapter::Action &action);
	void resetCommandReference();
	void resetState();
	bool updateVehicleState(float &dt_s);
	void updateConvertedState();
	bool anyAxisActive(const bool axes[3]) const;
	bool armStateValid() const;
	bool vehicleStateValid() const;
	bool attitudeValid() const;
	bool angularVelocityValid() const;
	bool trajectorySetpointValid() const;
	bool offboardControlModeFresh() const;
	bool offboardControlModeSupported() const;
	bool offboardControlModeValid() const;
	ActiveMode activeMode();

	uORB::SubscriptionInterval _parameter_update_sub{ORB_ID(parameter_update), 1_s};
	uORB::Subscription _register_ext_component_reply_sub{ORB_ID(register_ext_component_reply)};
	uORB::Subscription _arming_check_request_sub{ORB_ID(arming_check_request)};
	uORB::Subscription _offboard_control_mode_sub{ORB_ID(offboard_control_mode)};
	uORB::Subscription _vehicle_status_sub{ORB_ID(vehicle_status)};
	uORB::Subscription _position_sub{ORB_ID(vehicle_local_position)};
	uORB::Subscription _trajectory_setpoint_sub{ORB_ID(trajectory_setpoint)};
	uORB::Subscription _attitude_sub{ORB_ID(vehicle_attitude)};
	uORB::Subscription _arm_joint_state_sub{ORB_ID(arm_joint_state)};
	uORB::SubscriptionCallbackWorkItem _angular_velocity_sub{this, ORB_ID(vehicle_angular_velocity)};

	uORB::Publication<actuator_motors_s> _actuator_motors_pub{ORB_ID(actuator_motors)};
	uORB::Publication<register_ext_component_request_s> _register_ext_component_request_pub{ORB_ID(register_ext_component_request)};
	uORB::Publication<unregister_ext_component_s> _unregister_ext_component_pub{ORB_ID(unregister_ext_component)};
	uORB::Publication<vehicle_control_mode_s> _config_control_setpoints_pub{ORB_ID(config_control_setpoints)};
	uORB::Publication<arming_check_reply_s> _arming_check_reply_pub{ORB_ID(arming_check_reply)};

	perf_counter_t _loop_perf;
	hrt_abstime _last_run{0};
	hrt_abstime _last_arm_state_diag{0};
	hrt_abstime _last_manual_control_diag{0};
	hrt_abstime _last_offboard_diag{0};
	hrt_abstime _last_setpoint_diag{0};
	RegisteredMode _manual_mode;
	RegisteredMode _offboard_mode;
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
		(ParamBool<px4::params::AMPC_OFFB_EN>) _param_ampc_offb_en,
		(ParamFloat<px4::params::COM_OF_LOSS_T>) _param_com_of_loss_t
	)
};
