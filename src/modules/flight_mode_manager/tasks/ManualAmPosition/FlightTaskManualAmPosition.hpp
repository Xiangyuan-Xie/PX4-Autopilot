/****************************************************************************
 *
 * Flight task for AM Position manual setpoint generation.
 *
 ****************************************************************************/

#pragma once

#include <lib/mathlib/math/filter/AlphaFilter.hpp>
#include <lib/sticks/Sticks.hpp>
#include <lib/weather_vane/WeatherVane.hpp>

#include "../FlightTask/FlightTask.hpp"

class FlightTaskManualAmPosition : public FlightTask
{
public:
	FlightTaskManualAmPosition() = default;
	~FlightTaskManualAmPosition() override = default;

	bool activate(const trajectory_setpoint_s &last_setpoint) override;
	bool updateInitialize() override;
	bool update() override;

protected:
	void _setDefaultConstraints() override;
	bool _checkTakeoff() override;
	void _ekfResetHandlerHeading(float delta_psi) override;

private:
	static constexpr float kExpo{0.6f};
	static constexpr float kYawErrorTimeConstant{1.f};
	static constexpr float kYawErrorChangeThreshold{math::radians(1.f)};

	float getRollExpo() const;
	float getPitchExpo() const;
	float getYawExpo() const;
	float getThrottleZeroCenteredExpo() const;
	matrix::Vector2f getPitchRollExpo() const;

	void updateYawSetpoint();
	void updateConstraintsFromEstimator();
	void scaleSticks();
	void updateSetpoints();
	void updateXYLock();
	void updateAltitudeLock();
	void terrainFollowing(bool apply_brake, bool stopped);
	void respectMinAltitude();
	void respectMaxAltitude();
	void respectGroundSlowdown();
	bool updateYawCorrection(float yaw, float unaided_yaw, float dt_s);
	float updateYawLock(float yaw, float yawspeed_setpoint, float yaw_setpoint, float yaw_correction_prev) const;

	Sticks _sticks{this};
	WeatherVane _weathervane{this};

	bool _sticks_data_required{true};
	bool _terrain_hold{false};
	float _min_distance_to_ground{-INFINITY};
	float _max_distance_to_ground{INFINITY};
	float _dist_to_ground_lock{NAN};
	uint8_t _xy_reset_counter{0};
	uint8_t _z_reset_counter{0};

	AlphaFilter<float> _yawspeed_filter;
	float _yaw_error_ref{0.f};
	float _yaw_correction{0.f};
	bool _yaw_estimate_converging{false};
	AlphaFilter<float> _yaw_error_lpf{kYawErrorTimeConstant};

	DEFINE_PARAMETERS_CUSTOM_PARENT(FlightTask,
					(ParamInt<px4::params::AM_POS_MANL_CTRL>) _param_am_pos_manual_control,
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
					(ParamInt<px4::params::MPC_ALT_MODE>) _param_mpc_alt_mode,
					(ParamFloat<px4::params::MPC_Z_P>) _param_mpc_z_p,
					(ParamFloat<px4::params::MPC_LAND_ALT1>) _param_mpc_land_alt1,
					(ParamFloat<px4::params::MPC_LAND_ALT2>) _param_mpc_land_alt2,
					(ParamFloat<px4::params::MPC_LAND_SPEED>) _param_mpc_land_speed,
					(ParamFloat<px4::params::MPC_TKO_SPEED>) _param_mpc_tko_speed
				       )
};
