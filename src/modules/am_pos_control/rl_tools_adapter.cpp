/****************************************************************************
 *
 * rl_tools AM Position policy backend.
 *
 ****************************************************************************/

#include "rl_tools_adapter.hpp"

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
	if (_runtime_initialized) {
		rlt::reset(_device, rlt::checkpoint::actor::module, _policy_state, _rng);
	}
}

bool RlToolsAdapter::infer(const Observation &observation, Action &action)
{
	for (int i = 0; i < ObservationDim; ++i) {
		rlt::set(_device, _input, observation[i], 0, i);
	}

	rlt::evaluate_step(_device, rlt::checkpoint::actor::module, _input, _policy_state, _output, _policy_buffer, _rng, _mode);

	for (int i = 0; i < ActionDim; ++i) {
		action[i] = rlt::get(_device, _output, 0, i);
	}

	return true;
}
