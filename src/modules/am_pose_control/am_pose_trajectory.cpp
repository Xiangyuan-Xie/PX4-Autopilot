/****************************************************************************
 *
 * C2-continuous minimum-jerk pose reference for AM Pose.
 *
 ****************************************************************************/

#include "am_pose_trajectory.hpp"

#include <cmath>
#include <float.h>

#include <mathlib/mathlib.h>

namespace
{
constexpr float kVectorEpsilon{1e-5f};
constexpr int kMaximumDurationAdjustments{8};
constexpr float kRateLimitMargin{1.001f};
constexpr float kHoldDerivativeEpsilon{1e-4f};
}

void AmPoseTrajectory::solveQuintic(float p0, float v0, float a0, float pf, float vf, float af,
				    float duration_s, float coefficients[6])
{
	const float t2 = duration_s * duration_s;
	const float t3 = t2 * duration_s;
	const float t4 = t3 * duration_s;
	const float t5 = t4 * duration_s;

	coefficients[0] = p0;
	coefficients[1] = v0;
	coefficients[2] = 0.5f * a0;
	coefficients[3] = (20.f * (pf - p0) - (8.f * vf + 12.f * v0) * duration_s
			   - (3.f * a0 - af) * t2) / (2.f * t3);
	coefficients[4] = (30.f * (p0 - pf) + (14.f * vf + 16.f * v0) * duration_s
			   + (3.f * a0 - 2.f * af) * t2) / (2.f * t4);
	coefficients[5] = (12.f * (pf - p0) - (6.f * vf + 6.f * v0) * duration_s
			   - (a0 - af) * t2) / (2.f * t5);
}

AmPoseTrajectory::KinematicSample AmPoseTrajectory::evaluate(const float coefficients[6], float time_s)
{
	const float t2 = time_s * time_s;
	const float t3 = t2 * time_s;
	const float t4 = t3 * time_s;
	const float t5 = t4 * time_s;

	KinematicSample sample{};
	sample.position = coefficients[0] + coefficients[1] * time_s + coefficients[2] * t2
			  + coefficients[3] * t3 + coefficients[4] * t4 + coefficients[5] * t5;
	sample.velocity = coefficients[1] + 2.f * coefficients[2] * time_s + 3.f * coefficients[3] * t2
			  + 4.f * coefficients[4] * t3 + 5.f * coefficients[5] * t4;
	sample.acceleration = 2.f * coefficients[2] + 6.f * coefficients[3] * time_s
			      + 12.f * coefficients[4] * t2 + 20.f * coefficients[5] * t3;
	return sample;
}

void AmPoseTrajectory::velocityBezierControlPoints(const float coefficients[6], float duration_s,
		float control_points[5])
{
	const float duration_squared = duration_s * duration_s;
	const float duration_cubed = duration_squared * duration_s;
	const float duration_fourth = duration_cubed * duration_s;
	const float power[5] {
		coefficients[1],
		2.f *coefficients[2] *duration_s,
		3.f *coefficients[3] *duration_squared,
		4.f *coefficients[4] *duration_cubed,
		5.f *coefficients[5] *duration_fourth,
	};

	// The velocity polynomial is degree four in normalized time. Its Bezier
	// control points form a convex hull, providing a conservative rate bound
	// without sampling the segment.
	control_points[0] = power[0];
	control_points[1] = power[0] + power[1] / 4.f;
	control_points[2] = power[0] + power[1] / 2.f + power[2] / 6.f;
	control_points[3] = power[0] + 3.f * power[1] / 4.f + power[2] / 2.f + power[3] / 4.f;
	control_points[4] = power[0] + power[1] + power[2] + power[3] + power[4];
}

void AmPoseTrajectory::splitBezierMidpoint(const float control_points[5], float left[5], float right[5])
{
	float level_one[4] {};
	float level_two[3] {};
	float level_three[2] {};

	for (int index = 0; index < 4; ++index) {
		level_one[index] = 0.5f * (control_points[index] + control_points[index + 1]);
	}

	for (int index = 0; index < 3; ++index) {
		level_two[index] = 0.5f * (level_one[index] + level_one[index + 1]);
	}

	for (int index = 0; index < 2; ++index) {
		level_three[index] = 0.5f * (level_two[index] + level_two[index + 1]);
	}

	const float midpoint = 0.5f * (level_three[0] + level_three[1]);
	left[0] = control_points[0];
	left[1] = level_one[0];
	left[2] = level_two[0];
	left[3] = level_three[0];
	left[4] = midpoint;
	right[0] = midpoint;
	right[1] = level_three[1];
	right[2] = level_two[2];
	right[3] = level_one[3];
	right[4] = control_points[4];
}

void AmPoseTrajectory::velocityBounds(const float position_coefficients[3][6], const float yaw_coefficients[6],
				      float duration_s, float &maximum_velocity, float &maximum_yaw_rate)
{
	float velocity_left[3][5] {};
	float velocity_right[3][5] {};
	float yaw_rate_control_points[5] {};
	float yaw_rate_left[5] {};
	float yaw_rate_right[5] {};

	for (int axis = 0; axis < 3; ++axis) {
		float control_points[5] {};
		velocityBezierControlPoints(position_coefficients[axis], duration_s, control_points);
		splitBezierMidpoint(control_points, velocity_left[axis], velocity_right[axis]);
	}

	velocityBezierControlPoints(yaw_coefficients, duration_s, yaw_rate_control_points);
	splitBezierMidpoint(yaw_rate_control_points, yaw_rate_left, yaw_rate_right);
	maximum_velocity = 0.f;
	maximum_yaw_rate = 0.f;

	for (int half = 0; half < 2; ++half) {
		for (int control_point = 0; control_point < 5; ++control_point) {
			const matrix::Vector3f velocity {
				half == 0 ? velocity_left[0][control_point] : velocity_right[0][control_point],
				half == 0 ? velocity_left[1][control_point] : velocity_right[1][control_point],
				half == 0 ? velocity_left[2][control_point] : velocity_right[2][control_point],
			};
			const float yaw_rate = half == 0 ? yaw_rate_left[control_point] : yaw_rate_right[control_point];
			maximum_velocity = math::max(maximum_velocity, velocity.norm());
			maximum_yaw_rate = math::max(maximum_yaw_rate, fabsf(yaw_rate));
		}
	}
}

float AmPoseTrajectory::unwrapNear(float angle, float reference)
{
	return reference + matrix::wrap_pi(angle - reference);
}

void AmPoseTrajectory::setSegment(const Sample &start, const matrix::Vector3f &end_position,
				  const matrix::Vector3f &end_velocity, const matrix::Vector3f &end_acceleration,
				  float end_yaw, float end_yaw_rate, float end_yaw_acceleration,
				  float duration_s, bool hold_at_end, hrt_abstime now)
{
	_segment_duration_s = math::constrain(duration_s, kMinimumSegmentDurationS, kMaximumSegmentDurationS);
	_segment_start = now;
	_end_position = end_position;
	_end_yaw = matrix::wrap_pi(unwrapNear(end_yaw, start.yaw));
	_hold_at_end = hold_at_end;
	_reference_type = ReferenceType::Quintic;
	_velocity_last_update = 0;
	_waypoint_segment_count = 0;
	_waypoint_duration_s = 0.f;

	for (int axis = 0; axis < 3; ++axis) {
		solveQuintic(start.position(axis), start.velocity(axis), start.acceleration(axis),
			     end_position(axis), end_velocity(axis), end_acceleration(axis),
			     _segment_duration_s, _position_coefficients[axis]);
	}

	solveQuintic(start.yaw, start.yaw_rate, start.yaw_acceleration,
		     unwrapNear(end_yaw, start.yaw), end_yaw_rate, end_yaw_acceleration,
		     _segment_duration_s, _yaw_coefficients);
	_initialized = true;
}

void AmPoseTrajectory::reset(const matrix::Vector3f &position, float yaw, hrt_abstime now)
{
	Sample start{};
	start.position = position;
	start.yaw = yaw;
	setSegment(start, position, {}, {}, yaw, 0.f, 0.f, kMinimumSegmentDurationS, true, now);

	for (int axis = 0; axis < 3; ++axis) {
		_velocity_translation[axis].reset(0.f, 0.f, position(axis));
	}

	_velocity_yaw.reset(0.f, 0.f, yaw);
}

AmPoseTrajectory::Sample AmPoseTrajectory::sampleKinematics(hrt_abstime now, float offset_s) const
{
	Sample result{};

	if (!_initialized) {
		return result;
	}

	if (_reference_type == ReferenceType::ManualVelocity
	    || _reference_type == ReferenceType::OffboardVelocity) {
		VelocitySmoothing translation[3] {
			_velocity_translation[0], _velocity_translation[1], _velocity_translation[2]
		};
		VelocitySmoothing yaw = _velocity_yaw;
		float elapsed_s{0.f};

		if (now > _velocity_last_update) {
			elapsed_s = math::min((now - _velocity_last_update) * 1e-6f, kVelocityMaximumDtS);
		}

		const float sample_time_s = elapsed_s + math::max(offset_s, 0.f);

		for (int axis = 0; axis < 3; ++axis) {
			translation[axis].updateTraj(sample_time_s);
			result.position(axis) = translation[axis].getCurrentPosition();
			result.velocity(axis) = translation[axis].getCurrentVelocity();
			result.acceleration(axis) = translation[axis].getCurrentAcceleration();
		}

		yaw.updateTraj(sample_time_s);
		result.yaw = matrix::wrap_pi(yaw.getCurrentPosition());
		result.yaw_rate = yaw.getCurrentVelocity();
		result.yaw_acceleration = yaw.getCurrentAcceleration();
		return result;
	}

	if (_reference_type == ReferenceType::WaypointTrajectory) {
		const float elapsed_s = waypointElapsedTime(now) + math::max(offset_s, 0.f);

		if (elapsed_s >= _waypoint_duration_s) {
			result = _waypoint_end;

			if (!_waypoint_hold_at_end) {
				const float extrapolation_s = elapsed_s - _waypoint_duration_s;
				result.position += result.velocity * extrapolation_s
						   + result.acceleration * (0.5f * extrapolation_s * extrapolation_s);
				result.velocity += result.acceleration * extrapolation_s;
				result.yaw += result.yaw_rate * extrapolation_s
					      + 0.5f * result.yaw_acceleration * extrapolation_s * extrapolation_s;
				result.yaw_rate += result.yaw_acceleration * extrapolation_s;
			}

			result.yaw = matrix::wrap_pi(result.yaw);
			return result;
		}

		int segment_index = 0;

		while (segment_index + 1 < _waypoint_segment_count
		       && elapsed_s > _waypoint_segments[segment_index].start_time_s
		       + _waypoint_segments[segment_index].duration_s) {
			++segment_index;
		}

		const WaypointSegment &segment = _waypoint_segments[segment_index];
		const float segment_time_s = math::constrain(elapsed_s - segment.start_time_s, 0.f, segment.duration_s);

		for (int axis = 0; axis < 3; ++axis) {
			const KinematicSample axis_sample = evaluate(segment.position_coefficients[axis], segment_time_s);
			result.position(axis) = axis_sample.position;
			result.velocity(axis) = axis_sample.velocity;
			result.acceleration(axis) = axis_sample.acceleration;
		}

		const KinematicSample yaw_sample = evaluate(segment.yaw_coefficients, segment_time_s);
		result.yaw = matrix::wrap_pi(yaw_sample.position);
		result.yaw_rate = yaw_sample.velocity;
		result.yaw_acceleration = yaw_sample.acceleration;
		return result;
	}

	const float elapsed_s = math::max((now - _segment_start) * 1e-6f + offset_s, 0.f);

	if (_hold_at_end && elapsed_s >= _segment_duration_s) {
		result.position = _end_position;
		result.yaw = _end_yaw;
		return result;
	}

	const float bounded_elapsed_s = math::min(elapsed_s, _segment_duration_s);

	for (int axis = 0; axis < 3; ++axis) {
		const KinematicSample axis_sample = evaluate(_position_coefficients[axis], bounded_elapsed_s);
		result.position(axis) = axis_sample.position;
		result.velocity(axis) = axis_sample.velocity;
		result.acceleration(axis) = axis_sample.acceleration;
	}

	const KinematicSample yaw_sample = evaluate(_yaw_coefficients, bounded_elapsed_s);
	result.yaw = matrix::wrap_pi(yaw_sample.position);
	result.yaw_rate = yaw_sample.velocity;
	result.yaw_acceleration = yaw_sample.acceleration;
	return result;
}

void AmPoseTrajectory::replanPosition(const matrix::Vector3f &position, float yaw, hrt_abstime now,
				      float maximum_velocity, float maximum_yaw_rate)
{
	if (!_initialized) {
		reset(position, yaw, now);
	}

	const Sample start = sampleKinematics(now, 0.f);
	const float velocity_limit = math::max(maximum_velocity, 0.01f);
	const float yaw_rate_limit = math::max(maximum_yaw_rate, 0.01f);
	const float distance = (position - start.position).norm();
	const float yaw_distance = fabsf(matrix::wrap_pi(yaw - start.yaw));
	const float required_duration = math::max(kMinimumSegmentDurationS,
					math::max(kMinimumJerkPeakRate * distance / velocity_limit,
							kMinimumJerkPeakRate * yaw_distance / yaw_rate_limit));
	float duration_s = math::min(required_duration, kMaximumSegmentDurationS);
	matrix::Vector3f bounded_position{};
	float bounded_yaw_delta{0.f};

	// A retarget can begin with nonzero C2 derivatives. Extend the segment until the
	// candidate itself respects the configured rate limits, rather than only its endpoints.
	for (int adjustment = 0; adjustment < kMaximumDurationAdjustments; ++adjustment) {
		const float maximum_distance = velocity_limit * duration_s / kMinimumJerkPeakRate;
		bounded_position = position;

		if (distance > maximum_distance && distance > kVectorEpsilon) {
			bounded_position = start.position + (position - start.position) * (maximum_distance / distance);
		}

		const float maximum_yaw_distance = yaw_rate_limit * duration_s / kMinimumJerkPeakRate;
		bounded_yaw_delta = math::constrain(matrix::wrap_pi(yaw - start.yaw),
						    -maximum_yaw_distance, maximum_yaw_distance);
		float position_coefficients[3][6] {};
		float yaw_coefficients[6] {};

		for (int axis = 0; axis < 3; ++axis) {
			solveQuintic(start.position(axis), start.velocity(axis), start.acceleration(axis),
				     bounded_position(axis), 0.f, 0.f, duration_s, position_coefficients[axis]);
		}

		solveQuintic(start.yaw, start.yaw_rate, start.yaw_acceleration,
			     start.yaw + bounded_yaw_delta, 0.f, 0.f, duration_s, yaw_coefficients);
		float peak_velocity{0.f};
		float peak_yaw_rate{0.f};
		velocityBounds(position_coefficients, yaw_coefficients, duration_s, peak_velocity, peak_yaw_rate);

		const float rate_ratio = math::max(peak_velocity / velocity_limit, peak_yaw_rate / yaw_rate_limit);

		if (rate_ratio <= 1.f || duration_s >= kMaximumSegmentDurationS) {
			break;
		}

		duration_s = math::min(duration_s * rate_ratio * kRateLimitMargin, kMaximumSegmentDurationS);
	}

	setSegment(start, bounded_position, {}, {}, start.yaw + bounded_yaw_delta, 0.f, 0.f,
		   duration_s, true, now);
}

void AmPoseTrajectory::updateVelocity(const matrix::Vector3f &velocity, float yaw_rate, hrt_abstime now,
				      const VelocityConstraints &constraints, VelocitySource source)
{
	if (!_initialized) {
		return;
	}

	matrix::Vector3f velocity_command = velocity.isAllFinite() ? velocity : matrix::Vector3f{};
	const float yaw_rate_command = PX4_ISFINITE(yaw_rate) ? yaw_rate : 0.f;
	float dt_s{0.f};

	const ReferenceType reference_type = source == VelocitySource::Manual
					     ? ReferenceType::ManualVelocity : ReferenceType::OffboardVelocity;

	if (_reference_type != reference_type) {
		const Sample start = sampleKinematics(now, 0.f);

		for (int axis = 0; axis < 3; ++axis) {
			_velocity_translation[axis].reset(start.acceleration(axis), start.velocity(axis), start.position(axis));
		}

		_velocity_yaw.reset(start.yaw_acceleration, start.yaw_rate, start.yaw);
		_reference_type = reference_type;
		_waypoint_segment_count = 0;
		_waypoint_duration_s = 0.f;
		_velocity_last_update = now;

	} else if (now > _velocity_last_update) {
		dt_s = math::min((now - _velocity_last_update) * 1e-6f, kVelocityMaximumDtS);
	}

	for (int axis = 0; axis < 2; ++axis) {
		_velocity_translation[axis].updateTraj(dt_s);
		_velocity_translation[axis].setMaxJerk(math::max(constraints.horizontal_jerk, 0.01f));
		_velocity_translation[axis].setMaxAccel(math::max(constraints.horizontal_acceleration, 0.01f));
		// Manual speed is constrained by FlightTaskManualAmPose. Do not apply a
		// second limit here. Offboard commands are constrained before this call.
		_velocity_translation[axis].setMaxVel(FLT_MAX);
		_velocity_translation[axis].updateDurations(velocity_command(axis));
	}

	VelocitySmoothing::timeSynchronization(_velocity_translation, 2);

	_velocity_translation[2].updateTraj(dt_s);
	_velocity_yaw.updateTraj(dt_s);
	const float vertical_velocity_delta = velocity_command(2) - _velocity_translation[2].getCurrentVelocity();
	const float vertical_acceleration = vertical_velocity_delta >= 0.f
					    ? constraints.vertical_acceleration_up : constraints.vertical_acceleration_down;
	_velocity_translation[2].setMaxJerk(math::max(constraints.vertical_jerk, 0.01f));
	_velocity_translation[2].setMaxAccel(math::max(vertical_acceleration, 0.01f));
	_velocity_translation[2].setMaxVel(FLT_MAX);
	_velocity_translation[2].updateDurations(velocity_command(2));
	_velocity_yaw.setMaxJerk(math::max(constraints.yaw_jerk, 0.01f));
	_velocity_yaw.setMaxAccel(math::max(constraints.yaw_acceleration, 0.01f));
	_velocity_yaw.setMaxVel(FLT_MAX);
	_velocity_yaw.updateDurations(yaw_rate_command);
	_velocity_last_update = now;
}

AmPoseTrajectory::WaypointLoadResult AmPoseTrajectory::loadWaypoints(const Waypoint *waypoints, int count,
		hrt_abstime now, float maximum_velocity, float maximum_yaw_rate, bool hold_at_end)
{
	if (waypoints == nullptr || count < 1 || count > kMaximumWaypointCount) {
		return WaypointLoadResult::InvalidCount;
	}

	const float velocity_limit = math::max(maximum_velocity, 0.01f);
	const float yaw_rate_limit = math::max(maximum_yaw_rate, 0.01f);

	for (int index = 0; index < count; ++index) {
		const Waypoint &waypoint = waypoints[index];

		if (!waypoint.position.isAllFinite() || !waypoint.velocity.isAllFinite()
		    || !waypoint.acceleration.isAllFinite() || !PX4_ISFINITE(waypoint.yaw)
		    || !PX4_ISFINITE(waypoint.yaw_rate) || !PX4_ISFINITE(waypoint.yaw_acceleration)) {
			return WaypointLoadResult::NonFinite;
		}
	}

	const Waypoint &final_waypoint = waypoints[count - 1];

	if (hold_at_end && (final_waypoint.velocity.norm() > kHoldDerivativeEpsilon
			    || final_waypoint.acceleration.norm() > kHoldDerivativeEpsilon
			    || fabsf(final_waypoint.yaw_rate) > kHoldDerivativeEpsilon
			    || fabsf(final_waypoint.yaw_acceleration) > kHoldDerivativeEpsilon)) {
		return WaypointLoadResult::FinalStateNotHold;
	}

	Sample segment_start = sampleKinematics(now, 0.f);

	// Validate every candidate segment before mutating the active reference.
	for (int index = 0; index < count; ++index) {
		const Waypoint &waypoint = waypoints[index];
		float position_coefficients[3][6] {};
		float yaw_coefficients[6] {};
		const float end_yaw = unwrapNear(waypoint.yaw, segment_start.yaw);

		for (int axis = 0; axis < 3; ++axis) {
			solveQuintic(segment_start.position(axis), segment_start.velocity(axis), segment_start.acceleration(axis),
				     waypoint.position(axis), waypoint.velocity(axis), waypoint.acceleration(axis),
				     kWaypointIntervalS, position_coefficients[axis]);
		}

		solveQuintic(segment_start.yaw, segment_start.yaw_rate, segment_start.yaw_acceleration,
			     end_yaw, waypoint.yaw_rate, waypoint.yaw_acceleration, kWaypointIntervalS, yaw_coefficients);

		float peak_velocity{0.f};
		float peak_yaw_rate{0.f};
		velocityBounds(position_coefficients, yaw_coefficients, kWaypointIntervalS,
			       peak_velocity, peak_yaw_rate);

		if (peak_velocity > velocity_limit * kRateLimitMargin
		    || peak_yaw_rate > yaw_rate_limit * kRateLimitMargin) {
			return WaypointLoadResult::RateLimitExceeded;
		}

		segment_start.position = waypoint.position;
		segment_start.velocity = waypoint.velocity;
		segment_start.acceleration = waypoint.acceleration;
		segment_start.yaw = end_yaw;
		segment_start.yaw_rate = waypoint.yaw_rate;
		segment_start.yaw_acceleration = waypoint.yaw_acceleration;
	}

	segment_start = sampleKinematics(now, 0.f);

	for (int index = 0; index < count; ++index) {
		const Waypoint &waypoint = waypoints[index];
		WaypointSegment &segment = _waypoint_segments[index];
		segment.start_time_s = static_cast<float>(index) * kWaypointIntervalS;
		segment.duration_s = kWaypointIntervalS;
		const float end_yaw = unwrapNear(waypoint.yaw, segment_start.yaw);

		for (int axis = 0; axis < 3; ++axis) {
			solveQuintic(segment_start.position(axis), segment_start.velocity(axis), segment_start.acceleration(axis),
				     waypoint.position(axis), waypoint.velocity(axis), waypoint.acceleration(axis),
				     segment.duration_s, segment.position_coefficients[axis]);
		}

		solveQuintic(segment_start.yaw, segment_start.yaw_rate, segment_start.yaw_acceleration,
			     end_yaw, waypoint.yaw_rate, waypoint.yaw_acceleration,
			     segment.duration_s, segment.yaw_coefficients);
		segment_start.position = waypoint.position;
		segment_start.velocity = waypoint.velocity;
		segment_start.acceleration = waypoint.acceleration;
		segment_start.yaw = end_yaw;
		segment_start.yaw_rate = waypoint.yaw_rate;
		segment_start.yaw_acceleration = waypoint.yaw_acceleration;
	}

	_waypoint_end = segment_start;
	_waypoint_start = now;
	_waypoint_duration_s = static_cast<float>(count) * kWaypointIntervalS;
	_waypoint_segment_count = static_cast<uint8_t>(count);
	_waypoint_hold_at_end = hold_at_end;
	_reference_type = ReferenceType::WaypointTrajectory;
	_velocity_last_update = 0;
	_initialized = true;
	return WaypointLoadResult::Accepted;
}

void AmPoseTrajectory::cancelWaypoints(hrt_abstime now)
{
	if (!_initialized) {
		return;
	}

	const Sample current = sampleKinematics(now, 0.f);
	setSegment(current, current.position, {}, {}, current.yaw, 0.f, 0.f,
		   kMinimumSegmentDurationS, true, now);
}

void AmPoseTrajectory::shiftFrame(const matrix::Vector3f &position_delta, float yaw_delta)
{
	if (!_initialized || !position_delta.isAllFinite() || !PX4_ISFINITE(yaw_delta)) {
		return;
	}

	if (_reference_type == ReferenceType::ManualVelocity
	    || _reference_type == ReferenceType::OffboardVelocity) {
		for (int axis = 0; axis < 3; ++axis) {
			_velocity_translation[axis].setCurrentPosition(
				_velocity_translation[axis].getCurrentPosition() + position_delta(axis));
		}

		_velocity_yaw.setCurrentPosition(_velocity_yaw.getCurrentPosition() + yaw_delta);
		return;
	}

	if (_reference_type == ReferenceType::WaypointTrajectory) {
		for (int segment_index = 0; segment_index < _waypoint_segment_count; ++segment_index) {
			for (int axis = 0; axis < 3; ++axis) {
				_waypoint_segments[segment_index].position_coefficients[axis][0] += position_delta(axis);
			}

			_waypoint_segments[segment_index].yaw_coefficients[0] += yaw_delta;
		}

		_waypoint_end.position += position_delta;
		_waypoint_end.yaw += yaw_delta;
		return;
	}

	for (int axis = 0; axis < 3; ++axis) {
		_position_coefficients[axis][0] += position_delta(axis);
	}

	_yaw_coefficients[0] += yaw_delta;
	_end_position += position_delta;
	_end_yaw = matrix::wrap_pi(_end_yaw + yaw_delta);
}

bool AmPoseTrajectory::finished(hrt_abstime now) const
{
	if (!_initialized) {
		return true;
	}

	if (_reference_type == ReferenceType::ManualVelocity
	    || _reference_type == ReferenceType::OffboardVelocity) {
		return velocityAtRest();
	}

	if (_reference_type == ReferenceType::WaypointTrajectory) {
		return waypointElapsedTime(now) >= _waypoint_duration_s;
	}

	return now >= _segment_start + static_cast<hrt_abstime>(_segment_duration_s * 1'000'000.f);
}

float AmPoseTrajectory::waypointElapsedTime(hrt_abstime now) const
{
	if (_reference_type != ReferenceType::WaypointTrajectory || now <= _waypoint_start) {
		return 0.f;
	}

	return (now - _waypoint_start) * 1e-6f;
}

uint8_t AmPoseTrajectory::waypointSegmentIndex(hrt_abstime now) const
{
	if (_reference_type != ReferenceType::WaypointTrajectory || _waypoint_segment_count == 0) {
		return 0;
	}

	const float elapsed_s = waypointElapsedTime(now);
	uint8_t segment_index{0};

	while (segment_index + 1 < _waypoint_segment_count
	       && elapsed_s > _waypoint_segments[segment_index].start_time_s
	       + _waypoint_segments[segment_index].duration_s) {
		++segment_index;
	}

	return segment_index;
}

bool AmPoseTrajectory::velocityAtRest() const
{
	if (_reference_type != ReferenceType::ManualVelocity
	    && _reference_type != ReferenceType::OffboardVelocity) {
		return true;
	}

	for (int axis = 0; axis < 3; ++axis) {
		if (fabsf(_velocity_translation[axis].getCurrentVelocity()) > kVelocityRestEpsilon
		    || fabsf(_velocity_translation[axis].getCurrentAcceleration()) > kAccelerationRestEpsilon) {
			return false;
		}
	}

	return fabsf(_velocity_yaw.getCurrentVelocity()) <= kVelocityRestEpsilon
	       && fabsf(_velocity_yaw.getCurrentAcceleration()) <= kAccelerationRestEpsilon;
}

AmPoseTrajectory::Sample AmPoseTrajectory::sample(hrt_abstime now, float offset_s) const
{
	Sample result = sampleKinematics(now, offset_s);
	result.attitude = matrix::Dcmf(matrix::Eulerf(0.f, 0.f, result.yaw));
	return result;
}
