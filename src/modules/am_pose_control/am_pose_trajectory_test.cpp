#include <gtest/gtest.h>

#include <cmath>

#include "am_pose_trajectory.hpp"

using namespace time_literals;

namespace
{
constexpr hrt_abstime kStart{1_s};

AmPoseTrajectory::Waypoint waypoint(float x, float velocity_x = 0.f,
				    float acceleration_x = 0.f, float yaw = 0.f, float yaw_rate = 0.f,
				    float yaw_acceleration = 0.f)
{
	AmPoseTrajectory::Waypoint result{};
	result.position = {x, 0.f, 0.f};
	result.velocity = {velocity_x, 0.f, 0.f};
	result.acceleration = {acceleration_x, 0.f, 0.f};
	result.yaw = yaw;
	result.yaw_rate = yaw_rate;
	result.yaw_acceleration = yaw_acceleration;
	return result;
}
}

TEST(AmPoseTrajectoryTest, NearTargetReachesBeforeFirstPreviewAndHolds)
{
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, kStart);
	trajectory.replanPosition({0.02f, 0.f, 0.f}, 0.f, kStart, 1.f, 1.f);

	EXPECT_TRUE(trajectory.finished(kStart + 50_ms));

	for (int preview = 0; preview < AmPoseTrajectory::kPreviewCount; ++preview) {
		const auto target = trajectory.sample(kStart, AmPoseTrajectory::previewOffset(preview));
		EXPECT_NEAR(target.position(0), 0.02f, 1e-6f);
		EXPECT_NEAR(target.position(1), 0.f, 1e-6f);
		EXPECT_NEAR(target.position(2), 0.f, 1e-6f);
		EXPECT_NEAR(target.velocity.norm(), 0.f, 1e-6f);
		EXPECT_NEAR(target.acceleration.norm(), 0.f, 1e-6f);
		EXPECT_NEAR(target.yaw_rate, 0.f, 1e-6f);
	}
}

TEST(AmPoseTrajectoryTest, PositionReplanRespectsLinearAndYawRateLimits)
{
	constexpr float maximum_velocity{0.8f};
	constexpr float maximum_yaw_rate{0.6f};
	constexpr float duration_s{0.9375f};
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, kStart);
	trajectory.replanPosition({0.4f, 0.f, 0.f}, 0.3f, kStart, maximum_velocity, maximum_yaw_rate);

	for (int sample = 0; sample <= 100; ++sample) {
		const auto reference = trajectory.sample(kStart, duration_s * static_cast<float>(sample) / 100.f);
		EXPECT_LE(reference.velocity.norm(), maximum_velocity + 1e-5f);
		EXPECT_LE(fabsf(reference.yaw_rate), maximum_yaw_rate + 1e-5f);
	}
}

TEST(AmPoseTrajectoryTest, PositionReplanPreservesBoundaryState)
{
	AmPoseTrajectory trajectory{};
	trajectory.reset({0.f, 0.f, 0.f}, 0.f, kStart);
	const AmPoseTrajectory::VelocityConstraints constraints{3.f, 8.f, 3.f, 3.f, 8.f, 3.f, 8.f};

	for (int tick = 0; tick <= 50; ++tick) {
		trajectory.updateVelocity({0.8f, 0.f, 0.f}, 0.3f,
					  kStart + static_cast<hrt_abstime>(tick) * 10_ms, constraints,
					  AmPoseTrajectory::VelocitySource::Manual);
	}

	const hrt_abstime replan_time = kStart + 500_ms;
	const auto before = trajectory.sample(replan_time, 0.f);
	trajectory.replanPosition({1.f, 1.f, 0.f}, 0.8f, replan_time, 1.f, 1.f);
	const auto after = trajectory.sample(replan_time, 0.f);

	for (int axis = 0; axis < 3; ++axis) {
		EXPECT_NEAR(after.position(axis), before.position(axis), 1e-6f);
		EXPECT_NEAR(after.velocity(axis), before.velocity(axis), 1e-6f);
		EXPECT_NEAR(after.acceleration(axis), before.acceleration(axis), 1e-5f);
	}

	EXPECT_NEAR(after.yaw, before.yaw, 1e-6f);
	EXPECT_NEAR(after.yaw_rate, before.yaw_rate, 1e-6f);
	EXPECT_NEAR(after.yaw_acceleration, before.yaw_acceleration, 1e-5f);
}

TEST(AmPoseTrajectoryTest, PositionReplanFromMotionRespectsRateLimits)
{
	constexpr float maximum_velocity{1.f};
	constexpr float maximum_yaw_rate{1.f};
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, kStart);
	const AmPoseTrajectory::VelocityConstraints constraints{3.f, 8.f, 3.f, 3.f, 8.f, 3.f, 8.f};

	for (int tick = 0; tick <= 100; ++tick) {
		trajectory.updateVelocity({0.8f, 0.f, 0.f}, 0.8f,
					  kStart + static_cast<hrt_abstime>(tick) * 10_ms, constraints,
					  AmPoseTrajectory::VelocitySource::Manual);
	}

	const hrt_abstime replan_time = kStart + 1_s;
	trajectory.replanPosition({}, -0.4f, replan_time, maximum_velocity, maximum_yaw_rate);

	for (int sample = 0; sample <= 600; ++sample) {
		const auto reference = trajectory.sample(replan_time, 0.01f * static_cast<float>(sample));
		EXPECT_LE(reference.velocity.norm(), maximum_velocity + 1e-4f);
		EXPECT_LE(fabsf(reference.yaw_rate), maximum_yaw_rate + 1e-4f);
	}
}

TEST(AmPoseTrajectoryTest, PoseTargetsRemainYawOnlyDuringLateralMotion)
{
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, kStart);
	trajectory.replanPosition({1.f, 1.f, 0.f}, 0.8f, kStart, 1.f, 1.f);

	const auto reference = trajectory.sample(kStart, 0.4f);
	const matrix::Eulerf euler(reference.attitude);
	EXPECT_GT(reference.acceleration.norm(), 0.01f);
	EXPECT_NEAR(euler.phi(), 0.f, 1e-6f);
	EXPECT_NEAR(euler.theta(), 0.f, 1e-6f);
	EXPECT_NEAR(euler.psi(), reference.yaw, 1e-6f);
}

TEST(AmPoseTrajectoryTest, PreviewOffsetsMatchAmPoseTrainer)
{
	EXPECT_FLOAT_EQ(AmPoseTrajectory::previewOffset(0), 0.05f);

	for (int preview = 1; preview < AmPoseTrajectory::kPreviewCount; ++preview) {
		EXPECT_FLOAT_EQ(AmPoseTrajectory::previewOffset(preview),
				AmPoseTrajectory::previewOffset(preview - 1) + 0.05f);
	}

	EXPECT_FLOAT_EQ(AmPoseTrajectory::previewOffset(AmPoseTrajectory::kPreviewCount - 1),
			AmPoseTrajectory::kPreviewHorizonS);
}

TEST(AmPoseTrajectoryTest, FarTargetCompletesIntermediateSegmentBeforeFinalTarget)
{
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, 1_s);
	trajectory.replanPosition({20.f, 0.f, 0.f}, 0.f, 1_s, 1.f, 1.f);

	EXPECT_FALSE(trajectory.finished(6999_ms));
	EXPECT_TRUE(trajectory.finished(7_s));

	const auto endpoint = trajectory.sample(7_s, 0.f);
	EXPECT_LT(endpoint.position(0), 20.f);
	EXPECT_NEAR(endpoint.velocity.norm(), 0.f, 1e-6f);
	EXPECT_NEAR(endpoint.acceleration.norm(), 0.f, 1e-6f);
}

TEST(AmPoseTrajectoryTest, ManualVelocityDoesNotApplyOffboardLimit)
{
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, kStart);
	const AmPoseTrajectory::VelocityConstraints constraints{3.f, 8.f, 3.f, 3.f, 8.f, 3.f, 8.f};
	AmPoseTrajectory::Sample previous{};

	for (int tick = 0; tick <= 250; ++tick) {
		const hrt_abstime now = kStart + static_cast<hrt_abstime>(tick) * 10_ms;
		trajectory.updateVelocity({1.5f, 0.f, 0.f}, 0.8f, now, constraints,
					  AmPoseTrajectory::VelocitySource::Manual);
		const auto reference = trajectory.sample(now, 0.f);

		EXPECT_LE(fabsf(reference.velocity(0)), 1.5f + 1e-4f);
		EXPECT_LE(fabsf(reference.acceleration(0)), constraints.horizontal_acceleration + 1e-4f);
		EXPECT_LE(fabsf(reference.yaw_rate), 0.8f + 1e-4f);
		EXPECT_LE(fabsf(reference.yaw_acceleration), constraints.yaw_acceleration + 1e-4f);

		if (tick > 0) {
			EXPECT_LE(fabsf(reference.acceleration(0) - previous.acceleration(0)) / 0.01f,
				  constraints.horizontal_jerk + 1e-3f);
			EXPECT_LE(fabsf(reference.yaw_acceleration - previous.yaw_acceleration) / 0.01f,
				  constraints.yaw_jerk + 1e-3f);
		}

		previous = reference;
	}

	const auto endpoint = trajectory.sample(kStart + 2500_ms, 0.f);
	EXPECT_NEAR(endpoint.velocity(0), 1.5f, 1e-3f);
	EXPECT_NEAR(endpoint.yaw_rate, 0.8f, 1e-3f);
}

TEST(AmPoseTrajectoryTest, ManualVelocityUsesIndependentVerticalAccelerationLimits)
{
	const AmPoseTrajectory::VelocityConstraints constraints{3.f, 8.f, 1.f, 2.f, 100.f, 3.f, 8.f};
	AmPoseTrajectory upward{};
	upward.reset({}, 0.f, kStart);
	float maximum_upward_acceleration{0.f};

	for (int tick = 0; tick <= 100; ++tick) {
		const hrt_abstime now = kStart + static_cast<hrt_abstime>(tick) * 10_ms;
		upward.updateVelocity({0.f, 0.f, 1.f}, 0.f, now, constraints,
				      AmPoseTrajectory::VelocitySource::Manual);
		maximum_upward_acceleration = fmaxf(maximum_upward_acceleration,
						    upward.sample(now, 0.f).acceleration(2));
	}

	EXPECT_NEAR(maximum_upward_acceleration, constraints.vertical_acceleration_up, 1e-3f);

	AmPoseTrajectory downward{};
	downward.reset({}, 0.f, kStart);
	float minimum_downward_acceleration{0.f};

	for (int tick = 0; tick <= 100; ++tick) {
		const hrt_abstime now = kStart + static_cast<hrt_abstime>(tick) * 10_ms;
		downward.updateVelocity({0.f, 0.f, -1.f}, 0.f, now, constraints,
					AmPoseTrajectory::VelocitySource::Manual);
		minimum_downward_acceleration = fminf(minimum_downward_acceleration,
						      downward.sample(now, 0.f).acceleration(2));
	}

	EXPECT_NEAR(minimum_downward_acceleration, -constraints.vertical_acceleration_down, 1e-3f);
}

TEST(AmPoseTrajectoryTest, ManualVelocityBrakesContinuouslyAndPreviewDoesNotMutateState)
{
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, kStart);
	const AmPoseTrajectory::VelocityConstraints constraints{3.f, 8.f, 3.f, 2.f, 8.f, 3.f, 8.f};

	for (int tick = 0; tick <= 100; ++tick) {
		trajectory.updateVelocity({1.5f, 0.75f, 1.f}, 0.8f,
					  kStart + static_cast<hrt_abstime>(tick) * 10_ms, constraints,
					  AmPoseTrajectory::VelocitySource::Manual);
	}

	const hrt_abstime brake_start = kStart + 1010_ms;
	const auto before_brake = trajectory.sample(brake_start, 0.f);

	for (int tick = 0; tick <= 250; ++tick) {
		const hrt_abstime now = brake_start + static_cast<hrt_abstime>(tick) * 10_ms;
		trajectory.updateVelocity({}, 0.f, now, constraints, AmPoseTrajectory::VelocitySource::Manual);
		const auto reference = trajectory.sample(now, 0.f);

		EXPECT_LE(fabsf(reference.acceleration(2)), constraints.vertical_acceleration_down + 1e-4f);
		EXPECT_LE(fabsf(reference.yaw_acceleration), constraints.yaw_acceleration + 1e-4f);
	}

	const hrt_abstime stopped_at = brake_start + 2500_ms;
	const auto stopped = trajectory.sample(stopped_at, 0.f);
	const auto preview = trajectory.sample(stopped_at, 0.5f);
	const auto after_preview = trajectory.sample(stopped_at, 0.f);

	EXPECT_GT(before_brake.velocity.norm(), 0.1f);
	EXPECT_TRUE(trajectory.velocityAtRest());
	EXPECT_NEAR(stopped.velocity.norm(), 0.f, 1e-3f);
	EXPECT_NEAR(stopped.yaw_rate, 0.f, 1e-3f);
	EXPECT_NEAR(after_preview.position(0), stopped.position(0), 1e-6f);
	EXPECT_NEAR(after_preview.position(1), stopped.position(1), 1e-6f);
	EXPECT_NEAR(after_preview.position(2), stopped.position(2), 1e-6f);
	EXPECT_GE(preview.position(0), stopped.position(0) - 1e-5f);
}

TEST(AmPoseTrajectoryTest, OffboardVelocityReplacesPointWithoutBreakingC2State)
{
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, kStart);
	trajectory.replanPosition({1.f, 0.5f, 0.f}, 0.5f, kStart, 1.f, 1.f);
	const hrt_abstime switch_time = kStart + 400_ms;
	const auto before = trajectory.sample(switch_time, 0.f);
	const AmPoseTrajectory::VelocityConstraints constraints{3.f, 8.f, 3.f, 3.f, 8.f, 3.f, 8.f};

	trajectory.updateVelocity({0.f, 0.8f, 0.f}, -0.4f, switch_time, constraints,
				  AmPoseTrajectory::VelocitySource::Offboard);
	const auto after = trajectory.sample(switch_time, 0.f);

	EXPECT_LT((after.position - before.position).norm(), 1e-6f);
	EXPECT_LT((after.velocity - before.velocity).norm(), 1e-6f);
	EXPECT_LT((after.acceleration - before.acceleration).norm(), 1e-5f);
	EXPECT_NEAR(after.yaw, before.yaw, 1e-6f);
	EXPECT_NEAR(after.yaw_rate, before.yaw_rate, 1e-6f);
	EXPECT_NEAR(after.yaw_acceleration, before.yaw_acceleration, 1e-5f);
}

TEST(AmPoseTrajectoryTest, OffboardVelocityUsesSharedLimitsAndBrakesToRest)
{
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, kStart);
	const AmPoseTrajectory::VelocityConstraints constraints{2.f, 6.f, 1.f, 1.5f, 5.f, 1.2f, 4.f};
	AmPoseTrajectory::Sample previous{};

	for (int tick = 0; tick <= 200; ++tick) {
		const hrt_abstime now = kStart + static_cast<hrt_abstime>(tick) * 10_ms;
		trajectory.updateVelocity({0.6f, 0.8f, 0.5f}, 0.7f, now, constraints,
					  AmPoseTrajectory::VelocitySource::Offboard);
		const auto reference = trajectory.sample(now, 0.f);
		EXPECT_LE(reference.velocity.norm(), 1.2f);
		EXPECT_LE(fabsf(reference.acceleration(0)), constraints.horizontal_acceleration + 1e-4f);
		EXPECT_LE(fabsf(reference.acceleration(1)), constraints.horizontal_acceleration + 1e-4f);
		EXPECT_LE(fabsf(reference.acceleration(2)), constraints.vertical_acceleration_up + 1e-4f);
		EXPECT_LE(fabsf(reference.yaw_acceleration), constraints.yaw_acceleration + 1e-4f);

		if (tick > 0) {
			EXPECT_LE(fabsf(reference.acceleration(0) - previous.acceleration(0)) / 0.01f,
				  constraints.horizontal_jerk + 1e-3f);
			EXPECT_LE(fabsf(reference.yaw_acceleration - previous.yaw_acceleration) / 0.01f,
				  constraints.yaw_jerk + 1e-3f);
		}

		previous = reference;
	}

	const hrt_abstime brake_start = kStart + 2010_ms;

	for (int tick = 0; tick <= 250; ++tick) {
		trajectory.updateVelocity({}, 0.f, brake_start + static_cast<hrt_abstime>(tick) * 10_ms,
					  constraints, AmPoseTrajectory::VelocitySource::Offboard);
	}

	const auto stopped = trajectory.sample(brake_start + 2500_ms, 0.f);
	EXPECT_TRUE(trajectory.velocityAtRest());
	EXPECT_NEAR(stopped.velocity.norm(), 0.f, 1e-3f);
	EXPECT_NEAR(stopped.yaw_rate, 0.f, 1e-3f);
}

TEST(AmPoseTrajectoryTest, EstimatorFrameShiftPreservesTrajectoryDerivatives)
{
	AmPoseTrajectory trajectory{};
	trajectory.reset({1.f, 2.f, 3.f}, 0.2f, kStart);
	trajectory.replanPosition({3.f, 4.f, 5.f}, 0.8f, kStart, 1.f, 1.f);
	const auto before = trajectory.sample(kStart + 1_s, 0.f);

	trajectory.shiftFrame({4.f, -2.f, 1.f}, -0.3f);
	const auto after = trajectory.sample(kStart + 1_s, 0.f);

	EXPECT_TRUE((after.position - before.position - matrix::Vector3f{4.f, -2.f, 1.f}).norm() < 1e-5f);
	EXPECT_TRUE((after.velocity - before.velocity).norm() < 1e-5f);
	EXPECT_TRUE((after.acceleration - before.acceleration).norm() < 1e-5f);
	EXPECT_NEAR(matrix::wrap_pi(after.yaw - before.yaw), -0.3f, 1e-5f);
}

TEST(AmPoseTrajectoryTest, WaypointCountAndFixedTimingAreValidated)
{
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, kStart);
	AmPoseTrajectory::Waypoint points[AmPoseTrajectory::kMaximumWaypointCount] {};

	EXPECT_EQ(trajectory.loadWaypoints(nullptr, 0, kStart, 10.f, 10.f),
		  AmPoseTrajectory::WaypointLoadResult::InvalidCount);
	EXPECT_EQ(trajectory.loadWaypoints(points, AmPoseTrajectory::kMaximumWaypointCount + 1,
					   kStart, 10.f, 10.f), AmPoseTrajectory::WaypointLoadResult::InvalidCount);

	points[0] = waypoint(0.01f);
	EXPECT_EQ(trajectory.loadWaypoints(points, 1, kStart, 10.f, 10.f),
		  AmPoseTrajectory::WaypointLoadResult::Accepted);
	EXPECT_NEAR(trajectory.waypointDuration(), AmPoseTrajectory::kWaypointIntervalS, 1e-6f);
}

TEST(AmPoseTrajectoryTest, FiveWaypointsInterpolateTenPreviewsAndHoldFinalState)
{
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, kStart);
	AmPoseTrajectory::Waypoint points[AmPoseTrajectory::kMaximumWaypointCount] {};

	for (int index = 0; index < AmPoseTrajectory::kMaximumWaypointCount; ++index) {
		points[index] = waypoint(0.05f * (index + 1));
	}

	ASSERT_EQ(trajectory.loadWaypoints(points, AmPoseTrajectory::kMaximumWaypointCount,
					   kStart, 2.f, 2.f), AmPoseTrajectory::WaypointLoadResult::Accepted);
	EXPECT_EQ(trajectory.waypointCount(), AmPoseTrajectory::kMaximumWaypointCount);
	EXPECT_EQ(trajectory.waypointSegmentIndex(kStart + 450_ms), 4);

	for (int preview = 0; preview < AmPoseTrajectory::kPreviewCount; ++preview) {
		const auto sample = trajectory.sample(kStart, AmPoseTrajectory::previewOffset(preview));
		EXPECT_TRUE(sample.position.isAllFinite());
	}

	for (int point = 0; point < AmPoseTrajectory::kMaximumWaypointCount; ++point) {
		const auto sample = trajectory.sample(kStart, AmPoseTrajectory::previewOffset(point * 2 + 1));
		EXPECT_NEAR(sample.position(0), points[point].position(0), 1e-5f);
	}

	const auto interpolated = trajectory.sample(kStart, AmPoseTrajectory::previewOffset(0));
	EXPECT_GT(interpolated.position(0), 0.f);
	EXPECT_LT(interpolated.position(0), points[0].position(0));

	const auto endpoint = trajectory.sample(kStart + 500_ms, 0.f);
	const auto held = trajectory.sample(kStart + 5_s, 0.5f);
	EXPECT_NEAR(endpoint.position(0), 0.25f, 1e-5f);
	EXPECT_NEAR(held.position(0), endpoint.position(0), 1e-6f);
	EXPECT_NEAR(held.velocity.norm(), 0.f, 1e-6f);
	EXPECT_NEAR(held.acceleration.norm(), 0.f, 1e-6f);
	EXPECT_TRUE(trajectory.finished(kStart + 500_ms));
}

TEST(AmPoseTrajectoryTest, WaypointBoundariesAreC2Continuous)
{
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, kStart);
	AmPoseTrajectory::Waypoint points[] {
		waypoint(0.02f, 0.3f, 0.1f, 0.15f, 0.2f, 0.05f),
		waypoint(0.04f, 0.f, 0.f, 0.3f),
	};

	ASSERT_EQ(trajectory.loadWaypoints(points, 2, kStart, 10.f, 10.f),
		  AmPoseTrajectory::WaypointLoadResult::Accepted);
	const auto boundary = trajectory.sample(kStart + 100_ms, 0.f);
	const auto before = trajectory.sample(kStart + 99999, 0.f);
	const auto after = trajectory.sample(kStart + 100001, 0.f);

	EXPECT_NEAR(boundary.position(0), points[0].position(0), 1e-5f);
	EXPECT_NEAR(boundary.velocity(0), points[0].velocity(0), 1e-5f);
	EXPECT_NEAR(boundary.acceleration(0), points[0].acceleration(0), 1e-4f);
	EXPECT_NEAR(boundary.yaw, points[0].yaw, 1e-5f);
	EXPECT_NEAR(boundary.yaw_rate, points[0].yaw_rate, 1e-5f);
	EXPECT_NEAR(boundary.yaw_acceleration, points[0].yaw_acceleration, 1e-4f);
	EXPECT_NEAR(before.position(0), after.position(0), 1e-5f);
	EXPECT_NEAR(before.velocity(0), after.velocity(0), 1e-5f);
	EXPECT_NEAR(before.acceleration(0), after.acceleration(0), 3e-4f);
}

TEST(AmPoseTrajectoryTest, InvalidReplacementLeavesActiveTrajectoryUntouched)
{
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, kStart);
	AmPoseTrajectory::Waypoint valid[] {waypoint(0.05f)};
	ASSERT_EQ(trajectory.loadWaypoints(valid, 1, kStart, 2.f, 2.f),
		  AmPoseTrajectory::WaypointLoadResult::Accepted);
	const auto before = trajectory.sample(kStart, 0.04f);

	AmPoseTrajectory::Waypoint invalid[] {waypoint(10.f)};
	EXPECT_EQ(trajectory.loadWaypoints(invalid, 1, kStart, 0.2f, 2.f),
		  AmPoseTrajectory::WaypointLoadResult::RateLimitExceeded);
	const auto after = trajectory.sample(kStart, 0.04f);

	EXPECT_NEAR(after.position(0), before.position(0), 1e-6f);
	EXPECT_EQ(trajectory.waypointCount(), 1);
}

TEST(AmPoseTrajectoryTest, CancelTransitionsAnyReferenceToHold)
{
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, kStart);
	trajectory.replanPosition({1.f, 0.f, 0.f}, 0.5f, kStart, 1.f, 1.f);
	const hrt_abstime cancel_time = kStart + 200_ms;
	const auto before_cancel = trajectory.sample(cancel_time, 0.f);

	trajectory.cancelWaypoints(cancel_time);
	const auto held = trajectory.sample(cancel_time + 1_s, 0.f);

	EXPECT_TRUE(trajectory.finished(cancel_time + 1_s));
	EXPECT_NEAR(held.position(0), before_cancel.position(0), 1e-5f);
	EXPECT_NEAR(held.position(1), before_cancel.position(1), 1e-5f);
	EXPECT_NEAR(held.position(2), before_cancel.position(2), 1e-5f);
	EXPECT_NEAR(held.yaw, before_cancel.yaw, 1e-5f);
	EXPECT_NEAR(held.velocity.norm(), 0.f, 1e-6f);
	EXPECT_NEAR(held.acceleration.norm(), 0.f, 1e-6f);
}

TEST(AmPoseTrajectoryTest, FinalWaypointMustBeAHoldState)
{
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, kStart);
	AmPoseTrajectory::Waypoint point[] {waypoint(0.05f, 0.1f)};

	EXPECT_EQ(trajectory.loadWaypoints(point, 1, kStart, 2.f, 2.f),
		  AmPoseTrajectory::WaypointLoadResult::FinalStateNotHold);
}

TEST(AmPoseTrajectoryTest, RollingWindowAllowsNonzeroFinalDerivatives)
{
	AmPoseTrajectory trajectory{};
	trajectory.reset({}, 0.f, kStart);
	AmPoseTrajectory::Waypoint points[] {
		waypoint(0.02f, 0.2f),
		waypoint(0.04f, 0.2f),
	};

	ASSERT_EQ(trajectory.loadWaypoints(points, 2, kStart, 1.f, 1.f, false),
		  AmPoseTrajectory::WaypointLoadResult::Accepted);
	EXPECT_FALSE(trajectory.waypointHoldsAtEnd());

	const auto endpoint = trajectory.sample(kStart + 200_ms, 0.f);
	const auto extrapolated = trajectory.sample(kStart + 250_ms, 0.f);
	EXPECT_NEAR(endpoint.velocity(0), 0.2f, 1e-5f);
	EXPECT_GT(extrapolated.position(0), endpoint.position(0));

	trajectory.cancelWaypoints(kStart + 200_ms);
	const auto held = trajectory.sample(kStart + 1_s, 0.f);
	EXPECT_NEAR(held.velocity.norm(), 0.f, 1e-6f);
	EXPECT_NEAR(held.acceleration.norm(), 0.f, 1e-6f);
}
