#include "UserModeIntention.hpp"

#include <gtest/gtest.h>

TEST(UserModeIntentionTest, AllowsDisarmedAmTestModeChange)
{
	EXPECT_TRUE(UserModeIntention::modeChangeAllowedByArmingState(false, vehicle_status_s::NAVIGATION_STATE_AM_TEST));
}

TEST(UserModeIntentionTest, RejectsArmedAmTestModeChange)
{
	EXPECT_FALSE(UserModeIntention::modeChangeAllowedByArmingState(true, vehicle_status_s::NAVIGATION_STATE_AM_TEST));
	EXPECT_TRUE(UserModeIntention::modeChangeAllowedByArmingState(true, vehicle_status_s::NAVIGATION_STATE_POSCTL));
}
