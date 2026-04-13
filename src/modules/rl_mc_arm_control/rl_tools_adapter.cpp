/****************************************************************************
 *
 * Minimal RAPTOR-style adapter shim.
 *
 ****************************************************************************/

#include "rl_tools_adapter.hpp"

bool RlToolsAdapter::init()
{
	reset();
	return true;
}

void RlToolsAdapter::reset()
{
	_last_control_step_us = 0;
	_last_native_step_us = 0;
	_intermediate_steps_since_native = 0;
	_last_action.fill(0.0f);
	rl_mc_arm_policy::reset(_hidden_state);
}

bool RlToolsAdapter::infer(uint64_t now_us, const Observation &observation, Action &action)
{
	if (_last_control_step_us != 0 && now_us - _last_control_step_us < kIntermediateStepUs) {
		action = _last_action;
		return true;
	}

	_last_control_step_us = now_us;

	const bool first_step = _last_native_step_us == 0;
	const bool native_due = !first_step && (now_us - _last_native_step_us >= kNativeStepUs);
	const bool force_native = !first_step && (_intermediate_steps_since_native + 1 >= kForceSyncNative);

	if (first_step || native_due || force_native) {
		rl_mc_arm_policy::evaluate(observation, _hidden_state, _last_action);
		_last_native_step_us = now_us;
		_intermediate_steps_since_native = 0;

	} else {
		auto temp_hidden = _hidden_state;
		rl_mc_arm_policy::evaluate(observation, temp_hidden, _last_action);
		++_intermediate_steps_since_native;
	}

	action = _last_action;
	return true;
}
