/****************************************************************************
 *
 * AM Position policy adapter facade.
 *
 ****************************************************************************/

#include "am_policy_adapter.hpp"

bool AmPolicyAdapter::init()
{
	return _backend.init();
}

void AmPolicyAdapter::reset()
{
	_last_control_step_us = 0;
	_last_native_step_us = 0;
	_intermediate_steps_since_native = 0;

	for (int i = 0; i < ActionDim; ++i) {
		_last_action[i] = 0.0f;
	}

	_backend.reset();
}

bool AmPolicyAdapter::infer(uint64_t now_us, const Observation &observation, Action &action)
{
	if (_last_control_step_us != 0 && now_us - _last_control_step_us < kIntermediateStepUs) {
		for (int i = 0; i < ActionDim; ++i) {
			action[i] = _last_action[i];
		}

		return true;
	}

	_last_control_step_us = now_us;

	const bool first_step = _last_native_step_us == 0;
	const bool native_due = !first_step && (now_us - _last_native_step_us >= kNativeStepUs);
	const bool force_native = !first_step && (_intermediate_steps_since_native + 1 >= kForceSyncNative);

	if (first_step || native_due || force_native) {
		if (!_backend.infer(observation, _last_action)) {
			return false;
		}

		_last_native_step_us = now_us;
		_intermediate_steps_since_native = 0;

	} else {
		++_intermediate_steps_since_native;
	}

	for (int i = 0; i < ActionDim; ++i) {
		action[i] = _last_action[i];
	}

	return true;
}
