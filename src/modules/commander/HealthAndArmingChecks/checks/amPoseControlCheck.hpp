#pragma once

#include "../Common.hpp"

#include <uORB/Subscription.hpp>
#include <uORB/topics/am_pose_control_status.h>
#include <uORB/topics/offboard_control_mode.h>

class AmPoseControlChecks : public HealthAndArmingCheckBase
{
public:
	AmPoseControlChecks() = default;
	~AmPoseControlChecks() = default;

	void checkAndReport(const Context &context, Report &reporter) override;

private:
	static constexpr hrt_abstime kStatusTimeout = 1_s;

	uORB::Subscription _am_pose_control_status_sub{ORB_ID(am_pose_control_status)};
	uORB::Subscription _offboard_control_mode_sub{ORB_ID(offboard_control_mode)};
};
