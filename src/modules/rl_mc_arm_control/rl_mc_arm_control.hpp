/****************************************************************************
 *
 * RL multicopter arm direct actuator controller.
 *
 ****************************************************************************/
#pragma once

#include <array>

#include <mathlib/mathlib.h>
#include <mathlib/math/Functions.hpp>
#include <lib/sticks/Sticks.hpp>
#include <perf/perf_counter.h>
#include <px4_platform_common/px4_config.h>
#include <px4_platform_common/defines.h>
#include <px4_platform_common/log.h>
#include <px4_platform_common/module.h>
#include <px4_platform_common/module_params.h>
#include <px4_platform_common/px4_work_queue/WorkItem.hpp>
#include <matrix/matrix/math.hpp>
#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/SubscriptionCallback.hpp>
#include <uORB/topics/actuator_motors.h>
#include <uORB/topics/arm_joint_command.h>
#include <uORB/topics/arm_joint_state.h>
#include <uORB/topics/arming_check_reply.h>
#include <uORB/topics/arming_check_request.h>
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

#include "rl_mc_arm_control_manual_reference.hpp"
#include "rl_tools_adapter.hpp"

using namespace time_literals;

class RlMcArmControl : public ModuleBase<RlMcArmControl>, public ModuleParams, public px4::WorkItem
{
public:
	RlMcArmControl();
	~RlMcArmControl() override;

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

	struct CommandReference {
		matrix::Vector3f desired_lin_vel_b{};
		matrix::Vector3f desired_ang_vel_b{};
		matrix::Vector3f desired_pos_w{};
		matrix::Quatf desired_quat_w{};
		std::array<bool, 3> lin_cmd_active{{false, false, false}};
		std::array<bool, 3> ang_cmd_active{{false, false, false}};
		bool has_lin_vel_cmd{false};
		bool has_ang_vel_cmd{false};
	};

	void Run() override;
	void registerMode();
	void unregisterMode();
	void configureMode(int8_t mode_id);
	void replyToArmingCheck(int8_t request_id);
	void checkModeRegistration();
	void updateTargets(float dt_s);
	void generateManualTargets(matrix::Vector3f &desired_lin_vel_b, matrix::Vector3f &desired_ang_vel_b,
				  matrix::Vector3f &desired_pos_w, matrix::Quatf &desired_quat_w, float dt_s);
	void buildObservation(RlToolsAdapter::Observation &observation);
	void applyAction(const RlToolsAdapter::Action &action);
	void updateActionHistory(const RlToolsAdapter::Action &action);
	void resetHoverLock();
	void resetManualReferenceState();
	void resetState();
	bool updateVehicleState(float &dt_s);
	void updateConvertedState();
	bool anyAxisActive(const std::array<bool, 3> &axes) const;

	uORB::SubscriptionInterval _parameter_update_sub{ORB_ID(parameter_update), 1_s};
	uORB::Subscription _register_ext_component_reply_sub{ORB_ID(register_ext_component_reply)};
	uORB::Subscription _arming_check_request_sub{ORB_ID(arming_check_request)};
	uORB::Subscription _vehicle_status_sub{ORB_ID(vehicle_status)};
	uORB::Subscription _position_sub{ORB_ID(vehicle_local_position)};
	uORB::Subscription _trajectory_setpoint_sub{ORB_ID(trajectory_setpoint)};
	uORB::Subscription _attitude_sub{ORB_ID(vehicle_attitude)};
	uORB::Subscription _arm_joint_command_sub{ORB_ID(arm_joint_command)};
	uORB::Subscription _arm_joint_state_sub{ORB_ID(arm_joint_state)};
	uORB::SubscriptionCallbackWorkItem _angular_velocity_sub{this, ORB_ID(vehicle_angular_velocity)};

	uORB::Publication<actuator_motors_s> _actuator_motors_pub{ORB_ID(actuator_motors)};
	uORB::Publication<register_ext_component_request_s> _register_ext_component_request_pub{ORB_ID(register_ext_component_request)};
	uORB::Publication<unregister_ext_component_s> _unregister_ext_component_pub{ORB_ID(unregister_ext_component)};
	uORB::Publication<vehicle_control_mode_s> _config_control_setpoints_pub{ORB_ID(config_control_setpoints)};
	uORB::Publication<arming_check_reply_s> _arming_check_reply_pub{ORB_ID(arming_check_reply)};

	perf_counter_t _loop_perf;
	hrt_abstime _last_run{0};
	uint8_t _mode_request_id{77};
	int8_t _arming_check_id{-1};
	int8_t _mode_id{-1};
	bool _sent_mode_registration{false};
	bool _use_rl_mode{false};
	bool _hover_lock_active{false};
	matrix::Vector3f _hover_lock_pos_w{};
	float _hover_lock_yaw_w{0.0f};
	CommandReference _current_cmd_ref{};

	trajectory_setpoint_s _trajectory_setpoint{};
	vehicle_angular_velocity_s _angular_velocity{};
	vehicle_local_position_s _position{};
	vehicle_attitude_s _attitude{};
	arm_joint_command_s _arm_joint_command{};
	arm_joint_state_s _arm_joint_state{};
	matrix::Vector3f _root_pos_w{};
	matrix::Quaternionf _root_quat_w{};
	matrix::Vector3f _root_lin_vel_w{};
	matrix::Vector3f _root_lin_vel_b{};
	matrix::Vector3f _root_ang_vel_b{};
	float _heading_w{0.0f};

	std::array<float, kActionDim> _prev_action{{0.f, 0.f, 0.f, 0.f}};
	RlToolsAdapter _adapter{};
	Sticks _sticks{this};
	RlMcArmControlManualReference _manual_reference{};
	bool _last_manual_control_valid{false};
	bool _last_manual_control_nonzero{false};
	bool _last_manual_hold_xy{false};
	bool _last_manual_hold_z{false};
	bool _last_manual_hold_yaw{false};
	float _last_raw_roll{0.0f};
	float _last_raw_pitch{0.0f};
	float _last_raw_yaw{0.0f};
	float _last_raw_throttle{0.0f};
	matrix::Vector3f _last_desired_pos_w{};
	matrix::Vector3f _last_desired_lin_vel_b{};
	matrix::Vector3f _last_desired_ang_vel_b{};
	float _last_processed_roll{0.0f};
	float _last_processed_pitch{0.0f};
	float _last_processed_yaw{0.0f};
	float _last_processed_throttle{0.0f};

	DEFINE_PARAMETERS(
		(ParamInt<px4::params::RL_ARM_MANL_CTRL>) _param_manual_control,
		(ParamFloat<px4::params::MAPC_VEL_MANUAL>) _param_mapc_vel_manual,
		(ParamFloat<px4::params::MAPC_VEL_SIDE>) _param_mapc_vel_side,
		(ParamFloat<px4::params::MAPC_VEL_BACK>) _param_mapc_vel_back,
		(ParamFloat<px4::params::MAPC_ACC_HOR>) _param_mapc_acc_hor,
		(ParamFloat<px4::params::MAPC_JERK_MAX>) _param_mapc_jerk_max,
		(ParamFloat<px4::params::MAPC_Z_VEL_UP>) _param_mapc_z_vel_up,
		(ParamFloat<px4::params::MAPC_Z_VEL_DN>) _param_mapc_z_vel_dn,
		(ParamFloat<px4::params::MAPC_MAN_Y_MAX>) _param_mapc_man_y_max,
		(ParamFloat<px4::params::MAPC_MAN_Y_TAU>) _param_mapc_man_y_tau,
		(ParamFloat<px4::params::MAPC_MAN_DZ>) _param_mapc_man_dz,
		(ParamFloat<px4::params::MAPC_HOLD_MAX_Z>) _param_mapc_hold_max_z,
		(ParamFloat<px4::params::MAPC_HOLD_MAX_XY>) _param_mapc_hold_max_xy
	)
};
