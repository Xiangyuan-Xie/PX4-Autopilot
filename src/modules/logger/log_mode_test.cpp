#include <gtest/gtest.h>

#include "logger.h"

#include <uORB/topics/vehicle_status.h>

using px4::logger::Logger;

TEST(LoggerLogModeTest, StartsDisarmedAmTestLogging)
{
	EXPECT_TRUE(Logger::should_start_file_log(Logger::LogMode::while_armed, false, false,
			vehicle_status_s::NAVIGATION_STATE_AM_TEST));
}

TEST(LoggerLogModeTest, DoesNotStartDisarmedDefaultMode)
{
	EXPECT_FALSE(Logger::should_start_file_log(Logger::LogMode::while_armed, false, false,
			vehicle_status_s::NAVIGATION_STATE_POSCTL));
}

TEST(LoggerLogModeTest, StartsWhenArmedInAnyMode)
{
	EXPECT_TRUE(Logger::should_start_file_log(Logger::LogMode::while_armed, false, true,
			vehicle_status_s::NAVIGATION_STATE_POSCTL));
}

TEST(LoggerLogModeTest, AmTestDoesNotLatchArmUntilShutdown)
{
	const bool arm_until_shutdown_latched = false;

	EXPECT_FALSE(Logger::should_start_file_log(Logger::LogMode::arm_until_shutdown, arm_until_shutdown_latched, false,
					  vehicle_status_s::NAVIGATION_STATE_POSCTL));
}

TEST(LoggerLogModeTest, AmTestStartsWithoutArmUntilShutdownLatch)
{
	const bool arm_until_shutdown_latched = false;

	EXPECT_TRUE(Logger::should_start_file_log(Logger::LogMode::arm_until_shutdown, arm_until_shutdown_latched, false,
			vehicle_status_s::NAVIGATION_STATE_AM_TEST));
	EXPECT_FALSE(Logger::arm_until_shutdown_latched_after_update(Logger::LogMode::arm_until_shutdown,
			arm_until_shutdown_latched, false));
}

TEST(LoggerLogModeTest, ArmUntilShutdownStaysOnAfterArming)
{
	const bool arm_until_shutdown_latched = true;

	EXPECT_TRUE(Logger::should_start_file_log(Logger::LogMode::arm_until_shutdown, arm_until_shutdown_latched, false,
					 vehicle_status_s::NAVIGATION_STATE_POSCTL));
}

TEST(LoggerLogModeTest, ArmedStateLatchesArmUntilShutdown)
{
	const bool arm_until_shutdown_latched = false;

	EXPECT_TRUE(Logger::arm_until_shutdown_latched_after_update(Logger::LogMode::arm_until_shutdown,
			arm_until_shutdown_latched, true));
}
