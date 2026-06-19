/****************************************************************************
 *
 * Dreamer AM Position policy backend.
 *
 ****************************************************************************/

#include "dreamer_adapter.hpp"

#if defined(AM_POS_CONTROL_USE_DREAMER_POLICY)
bool DreamerAdapter::init()
{
	reset();
	return true;
}

void DreamerAdapter::reset()
{
	dreamer_policy::reset(_state);
}

bool DreamerAdapter::infer(const Observation &observation, Action &action)
{
	dreamer_policy::infer(observation, _state, action);
	return true;
}
#endif // AM_POS_CONTROL_USE_DREAMER_POLICY
