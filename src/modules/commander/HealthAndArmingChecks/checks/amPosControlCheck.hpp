#pragma once

#include "../Common.hpp"

#include <uORB/Subscription.hpp>
#include <uORB/topics/am_pos_control_status.h>

class AmPosControlChecks : public HealthAndArmingCheckBase
{
public:
	AmPosControlChecks() = default;
	~AmPosControlChecks() = default;

	void checkAndReport(const Context &context, Report &reporter) override;

private:
	static constexpr hrt_abstime kStatusTimeout = 1_s;

	uORB::Subscription _am_pos_control_status_sub{ORB_ID(am_pos_control_status)};
};
