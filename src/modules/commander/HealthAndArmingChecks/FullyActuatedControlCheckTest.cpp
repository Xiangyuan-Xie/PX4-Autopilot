#include <gtest/gtest.h>

#define private public
#include "Common.hpp"
#include "checks/fullyActuatedControlCheck.hpp"
#undef private

#include <uORB/Publication.hpp>
#include <uORB/topics/fully_actuated_control_status.h>

namespace
{

bool canArmWithStatus(int32_t airframe, const fully_actuated_control_status_s *status)
{
	uORB::Publication<fully_actuated_control_status_s> status_pub{ORB_ID(fully_actuated_control_status)};

	if (status != nullptr) {
		status_pub.publish(*status);
	}

	vehicle_status_s vehicle_status{};
	vehicle_status.vehicle_type = vehicle_status_s::VEHICLE_TYPE_ROTARY_WING;
	Context context{vehicle_status};
	failsafe_flags_s failsafe_flags{};
	Report reporter{failsafe_flags, 0_s};
	reporter.prepare(vehicle_status.vehicle_type);
	FullyActuatedControlChecks check;
	check._param_ca_airframe.set(airframe);
	check.checkAndReport(context, reporter);
	reporter.finalize();
	return reporter.canArm(vehicle_status_s::NAVIGATION_STATE_POSCTL);
}

fully_actuated_control_status_s validStatus()
{
	fully_actuated_control_status_s status{};
	status.timestamp = hrt_absolute_time();
	status.enabled = true;
	status.config_valid = true;
	status.matrix_rank = 6;
	status.matrix_full_rank = true;
	status.hover_feasible = true;
	return status;
}

} // namespace

TEST(FullyActuatedControlCheckTest, ValidGeometryAllowsArming)
{
	const auto status = validStatus();
	EXPECT_TRUE(canArmWithStatus(CA_AIRFRAME_FULLY_ACTUATED_MULTIROTOR, &status));
}

TEST(FullyActuatedControlCheckTest, RankDeficientGeometryBlocksArming)
{
	auto status = validStatus();
	status.matrix_rank = 5;
	status.matrix_full_rank = false;
	EXPECT_FALSE(canArmWithStatus(CA_AIRFRAME_FULLY_ACTUATED_MULTIROTOR, &status));
}

TEST(FullyActuatedControlCheckTest, OrdinaryMultirotorDoesNotRequireStatus)
{
	EXPECT_TRUE(canArmWithStatus(0, nullptr));
}
