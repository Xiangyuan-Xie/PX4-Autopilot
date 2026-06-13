#include "UserModeIntention.hpp"

#include <gtest/gtest.h>

TEST(UserModeIntentionTest, AllowsArmedModeChanges)
{
	EXPECT_TRUE(UserModeIntention::modeChangeAllowedByArmingState(true, vehicle_status_s::NAVIGATION_STATE_POSCTL));
	EXPECT_TRUE(UserModeIntention::modeChangeAllowedByArmingState(true, vehicle_status_s::NAVIGATION_STATE_AM_POSITION));
	EXPECT_TRUE(UserModeIntention::modeChangeAllowedByArmingState(true, vehicle_status_s::NAVIGATION_STATE_AM_OFFBOARD));
}
