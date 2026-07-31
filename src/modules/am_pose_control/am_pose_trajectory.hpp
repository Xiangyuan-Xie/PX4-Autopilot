/****************************************************************************
 *
 * C2-continuous minimum-jerk pose reference for AM Pose.
 *
 ****************************************************************************/

#pragma once

#include <drivers/drv_hrt.h>
#include <motion_planning/VelocitySmoothing.hpp>
#include <matrix/matrix/math.hpp>
#include <px4_platform_common/defines.h>
#include <px4_platform_common/time.h>

class AmPoseTrajectory
{
public:
	static constexpr float kMinimumSegmentDurationS{0.05f};
	static constexpr float kMaximumSegmentDurationS{6.f};
	static constexpr float kMinimumJerkPeakRate{1.875f};
	static constexpr int kPreviewCount{10};
	static constexpr int kMaximumWaypointCount{5};
	static constexpr float kWaypointIntervalS{0.1f};
	static constexpr float kPreviewIntervalS{0.05f};
	static constexpr float kPreviewHorizonS{kPreviewCount * kPreviewIntervalS};

	struct Sample {
		matrix::Vector3f position{};
		matrix::Vector3f velocity{};
		matrix::Vector3f acceleration{};
		float yaw{0.f};
		float yaw_rate{0.f};
		float yaw_acceleration{0.f};
		matrix::Dcmf attitude{};
	};

	struct VelocityConstraints {
		float horizontal_acceleration{3.f};
		float horizontal_jerk{8.f};
		float vertical_acceleration_up{3.f};
		float vertical_acceleration_down{3.f};
		float vertical_jerk{8.f};
		float yaw_acceleration{3.f};
		float yaw_jerk{8.f};
	};

	enum class VelocitySource : uint8_t {
		Manual,
		Offboard,
	};

	struct Waypoint {
		matrix::Vector3f position{};
		matrix::Vector3f velocity{};
		matrix::Vector3f acceleration{};
		float yaw{0.f};
		float yaw_rate{0.f};
		float yaw_acceleration{0.f};
	};

	enum class WaypointLoadResult : uint8_t {
		Accepted,
		InvalidCount,
		NonFinite,
		FinalStateNotHold,
		RateLimitExceeded,
	};

	void reset(const matrix::Vector3f &position, float yaw, hrt_abstime now);
	void replanPosition(const matrix::Vector3f &position, float yaw, hrt_abstime now,
			    float maximum_velocity, float maximum_yaw_rate);
	void updateVelocity(const matrix::Vector3f &velocity, float yaw_rate, hrt_abstime now,
			    const VelocityConstraints &constraints, VelocitySource source);
	WaypointLoadResult loadWaypoints(const Waypoint *waypoints, int count, hrt_abstime now,
					 float maximum_velocity, float maximum_yaw_rate, bool hold_at_end = true);
	void cancelWaypoints(hrt_abstime now);
	void shiftFrame(const matrix::Vector3f &position_delta, float yaw_delta);

	Sample sample(hrt_abstime now, float offset_s) const;
	bool initialized() const { return _initialized; }
	bool finished(hrt_abstime now) const;
	bool velocityAtRest() const;
	bool waypointTrajectoryActive() const { return _reference_type == ReferenceType::WaypointTrajectory; }
	uint8_t waypointCount() const { return _waypoint_segment_count; }
	uint8_t waypointSegmentIndex(hrt_abstime now) const;
	float waypointElapsedTime(hrt_abstime now) const;
	float waypointDuration() const { return _waypoint_duration_s; }
	bool waypointHoldsAtEnd() const { return _waypoint_hold_at_end; }

	static float previewOffset(int index)
	{
		return (index + 1) * kPreviewIntervalS;
	}

private:
	enum class ReferenceType : uint8_t {
		Quintic,
		ManualVelocity,
		OffboardVelocity,
		WaypointTrajectory,
	};

	struct KinematicSample {
		float position{0.f};
		float velocity{0.f};
		float acceleration{0.f};
	};

	struct WaypointSegment {
		float position_coefficients[3][6] {};
		float yaw_coefficients[6] {};
		float start_time_s{0.f};
		float duration_s{kMinimumSegmentDurationS};
	};

	static void solveQuintic(float start_position, float start_velocity, float start_acceleration,
				 float end_position, float end_velocity, float end_acceleration,
				 float duration_s, float coefficients[6]);
	static KinematicSample evaluate(const float coefficients[6], float time_s);
	static void velocityBezierControlPoints(const float coefficients[6], float duration_s, float control_points[5]);
	static void splitBezierMidpoint(const float control_points[5], float left[5], float right[5]);
	static void velocityBounds(const float position_coefficients[3][6], const float yaw_coefficients[6],
				   float duration_s, float &maximum_velocity, float &maximum_yaw_rate);
	static float unwrapNear(float angle, float reference);
	Sample sampleKinematics(hrt_abstime now, float offset_s) const;
	void setSegment(const Sample &start, const matrix::Vector3f &end_position,
			const matrix::Vector3f &end_velocity, const matrix::Vector3f &end_acceleration,
			float end_yaw, float end_yaw_rate, float end_yaw_acceleration,
			float duration_s, bool hold_at_end, hrt_abstime now);

	static constexpr float kVelocityMaximumDtS{0.02f};
	static constexpr float kVelocityRestEpsilon{0.01f};
	static constexpr float kAccelerationRestEpsilon{0.02f};

	float _position_coefficients[3][6] {};
	float _yaw_coefficients[6] {};
	hrt_abstime _segment_start{0};
	float _segment_duration_s{kMinimumSegmentDurationS};
	matrix::Vector3f _end_position{};
	float _end_yaw{0.f};
	bool _hold_at_end{true};
	bool _initialized{false};
	ReferenceType _reference_type{ReferenceType::Quintic};
	VelocitySmoothing _velocity_translation[3] {};
	VelocitySmoothing _velocity_yaw{};
	hrt_abstime _velocity_last_update{0};
	WaypointSegment _waypoint_segments[kMaximumWaypointCount] {};
	Sample _waypoint_end{};
	hrt_abstime _waypoint_start{0};
	float _waypoint_duration_s{0.f};
	uint8_t _waypoint_segment_count{0};
	bool _waypoint_hold_at_end{true};
};
