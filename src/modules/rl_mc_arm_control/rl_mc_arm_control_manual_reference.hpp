/****************************************************************************
 *
 * Position-style manual reference generator for rl_mc_arm_control.
 *
 ****************************************************************************/
#pragma once

#include <array>

#include <lib/mathlib/math/filter/AlphaFilter.hpp>
#include <matrix/matrix/math.hpp>
#include <motion_planning/ManualVelocitySmoothingXY.hpp>
#include <motion_planning/ManualVelocitySmoothingZ.hpp>

class RlMcArmControlManualReference
{
public:
	struct Params {
		float vel_manual{0.0f};
		float vel_side{-1.0f};
		float vel_back{-1.0f};
		float acc_hor{0.0f};
		float jerk_max{0.0f};
		float z_vel_up{0.0f};
		float z_vel_down{0.0f};
		float man_y_max_rad{0.0f};
		float man_y_tau{0.0f};
		float man_deadzone{0.0f};
		float hold_max_xy{0.0f};
		float hold_max_z{0.0f};
	};

	struct Input {
		float dt_s{0.0f};
		bool sticks_available{false};
		float roll{0.0f};
		float pitch{0.0f};
		float yaw{0.0f};
		float throttle{0.0f};
		matrix::Vector3f root_pos_w{};
		matrix::Quatf root_quat_w{};
		matrix::Vector3f root_lin_vel_w{};
		float heading_w{0.0f};
		float position_z_ned{0.0f};
		float velocity_z_ned{0.0f};
	};

	struct Output {
		matrix::Vector3f desired_lin_vel_b{};
		matrix::Vector3f desired_ang_vel_b{};
		matrix::Vector3f desired_pos_w{};
		matrix::Quatf desired_quat_w{};
		std::array<bool, 3> lin_cmd_active{{false, false, false}};
		std::array<bool, 3> ang_cmd_active{{false, false, false}};
		bool hold_xy_active{false};
		bool hold_z_active{false};
		bool hold_yaw_active{false};
		float processed_roll{0.0f};
		float processed_pitch{0.0f};
		float processed_yaw{0.0f};
		float processed_throttle{0.0f};
	};

	void reset();
	Output update(const Params &params, const Input &input);

private:
	void initializeIfNeeded(const Input &input);

	bool _initialized{false};
	ManualVelocitySmoothingXY _xy_smoothing{};
	ManualVelocitySmoothingZ _z_smoothing{};
	AlphaFilter<float> _yawspeed_filter{};
	matrix::Vector2f _hold_pos_xy{matrix::Vector2f(NAN, NAN)};
	float _yaw_sp{NAN};
};
