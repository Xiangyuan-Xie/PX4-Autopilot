/****************************************************************************
 *
 * Minimal RAPTOR-style adapter shim.
 *
 ****************************************************************************/

#include "rl_tools_adapter.hpp"

#include <cstring>

bool RlToolsAdapter::init()
{
	if (!_runtime_initialized) {
		rlt::init(_device);
		rlt::init(_device, _rng, 0);
		_runtime_initialized = true;
	}

	reset();
	return true;
}

void RlToolsAdapter::reset()
{
	_last_control_step_us = 0;
	_last_native_step_us = 0;
	_intermediate_steps_since_native = 0;
	for (int i = 0; i < ActionDim; ++i) {
		_last_action[i] = 0.0f;
	}
	if (_runtime_initialized) {
		rlt::reset(_device, rlt::checkpoint::actor::module, _policy_state, _rng);
	}
}

bool RlToolsAdapter::infer(uint64_t now_us, const Observation &observation, Action &action)
{
	if (_last_control_step_us != 0 && now_us - _last_control_step_us < kIntermediateStepUs) {
		for (int i = 0; i < ActionDim; ++i) {
			action[i] = _last_action[i];
		}
		return true;
	}

	_last_control_step_us = now_us;

	for (int i = 0; i < ObservationDim; ++i) {
		rlt::set(_device, _input, observation[i], 0, i);
	}

	const bool first_step = _last_native_step_us == 0;
	const bool native_due = !first_step && (now_us - _last_native_step_us >= kNativeStepUs);
	const bool force_native = !first_step && (_intermediate_steps_since_native + 1 >= kForceSyncNative);

	if (first_step || native_due || force_native) {
		rlt::evaluate_step(_device, rlt::checkpoint::actor::module, _input, _policy_state, _output, _policy_buffer, _rng, _mode);
		for (int i = 0; i < ActionDim; ++i) {
			_last_action[i] = rlt::get(_device, _output, 0, i);
		}
		_last_native_step_us = now_us;
		_intermediate_steps_since_native = 0;

	} else {
		memcpy(&_policy_state_temp, &_policy_state, sizeof(_policy_state));
		rlt::evaluate_step(_device, rlt::checkpoint::actor::module, _input, _policy_state_temp, _output, _policy_buffer, _rng, _mode);
		for (int i = 0; i < ActionDim; ++i) {
			_last_action[i] = rlt::get(_device, _output, 0, i);
		}
		++_intermediate_steps_since_native;
	}

	for (int i = 0; i < ActionDim; ++i) {
		action[i] = _last_action[i];
	}
	return true;
}
