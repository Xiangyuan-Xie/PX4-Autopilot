/****************************************************************************
 *
 * Dreamer AM Position policy backend.
 *
 ****************************************************************************/
#pragma once

#include "policy_types.hpp"

#if defined(AM_POS_CONTROL_USE_DREAMER_POLICY)
#include "blob/policy.h"

static_assert(dreamer_policy::ObservationDim == kPolicyObservationDim, "Dreamer policy observation dim mismatch");
static_assert(dreamer_policy::ActionDim == kPolicyActionDim, "Dreamer policy action dim mismatch");
static_assert(dreamer_policy::Deployable, "Dreamer policy blob must be deployable");

class DreamerAdapter
{
public:
	static constexpr int ObservationDim = dreamer_policy::ObservationDim;
	static constexpr int ActionDim = dreamer_policy::ActionDim;

	using Observation = PolicyObservation;
	using Action = PolicyAction;

	bool init();
	void reset();
	bool infer(const Observation &observation, Action &action);

private:
	dreamer_policy::State _state{};
};
#endif // AM_POS_CONTROL_USE_DREAMER_POLICY
