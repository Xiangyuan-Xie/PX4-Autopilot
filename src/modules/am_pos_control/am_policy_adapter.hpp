/****************************************************************************
 *
 * AM Position policy adapter facade.
 *
 ****************************************************************************/
#pragma once

#include <cstdint>

#include "policy_types.hpp"

#if defined(AM_POS_CONTROL_USE_DREAMER_POLICY)
#include "dreamer_adapter.hpp"
using AmPolicyBackend = DreamerAdapter;
#else
#include "rl_tools_adapter.hpp"
using AmPolicyBackend = RlToolsAdapter;
#endif

class AmPolicyAdapter
{
public:
	static constexpr int ObservationDim = kPolicyObservationDim;
	static constexpr int ActionDim = kPolicyActionDim;

	using Observation = PolicyObservation;
	using Action = PolicyAction;

	bool init();
	void reset();
	bool infer(uint64_t now_us, const Observation &observation, Action &action);

private:
	// Mirror the training control cadence: commit policy/RNN state at 100 Hz and hold the
	// committed action between native steps. Keep this cadence in the facade so PPO and
	// Dreamer backends share identical deployment timing semantics.
	static constexpr uint64_t kIntermediateStepUs = 2'500;
	static constexpr uint64_t kNativeStepUs = 10'000;
	static constexpr uint8_t kForceSyncNative = 4;

	uint64_t _last_control_step_us{0};
	uint64_t _last_native_step_us{0};
	uint8_t _intermediate_steps_since_native{0};
	Action _last_action{0.f, 0.f, 0.f, 0.f};
	AmPolicyBackend _backend{};
};
