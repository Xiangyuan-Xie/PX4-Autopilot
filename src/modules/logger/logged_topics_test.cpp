#include <gtest/gtest.h>

#include "logged_topics.h"

#include <uORB/topics/am_pos_control_status.h>
#include <uORB/topics/am_policy_observation.h>
#include <uORB/topics/am_test_result.h>
#include <uORB/topics/am_test_status.h>
#include <uORB/topics/arm_joint_state.h>
#include <uORB/topics/neural_control.h>
#include <uORB/uORBManager.hpp>

using px4::logger::LoggedTopics;
using px4::logger::SDLogProfileMask;

class LoggedTopicsTest : public ::testing::Test
{
	void SetUp() override
	{
		uORB::Manager::initialize();
	}

	void TearDown() override
	{
		uORB::Manager::terminate();
	}
};

TEST_F(LoggedTopicsTest, DefaultProfileLogsAmTopics)
{
	LoggedTopics topics;

	ASSERT_TRUE(topics.initialize_logged_topics(SDLogProfileMask::DEFAULT));

	EXPECT_TRUE(topics.has_subscription(ORB_ID(am_pos_control_status)));
	EXPECT_TRUE(topics.has_subscription(ORB_ID(am_policy_observation)));
	EXPECT_TRUE(topics.has_subscription(ORB_ID(am_test_result)));
	EXPECT_TRUE(topics.has_subscription(ORB_ID(am_test_status)));
	EXPECT_TRUE(topics.has_subscription(ORB_ID(arm_joint_state)));
	EXPECT_TRUE(topics.has_subscription(ORB_ID(neural_control)));
}
