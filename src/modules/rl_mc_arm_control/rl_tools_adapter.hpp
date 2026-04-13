/****************************************************************************
 *
 * Minimal RAPTOR-style adapter shim.
 *
 ****************************************************************************/
#pragma once

#include <array>
#include <cstdint>

#include "blob/policy.h"

class RlToolsAdapter
{
public:
	static constexpr int ObservationDim = static_cast<int>(rl_mc_arm_policy::kObservationDim);
	static constexpr int ActionDim = static_cast<int>(rl_mc_arm_policy::kActionDim);

	using Observation = std::array<float, ObservationDim>;
	using Action = std::array<float, ActionDim>;

	bool init();
	void reset();
	bool infer(uint64_t now_us, const Observation &observation, Action &action);

private:
	// Mirror the RAPTOR/L2F timing model locally: run a committed/native policy step at
	// 100 Hz, allow intermediate evaluations in between, and keep the true hidden state
	// unchanged on intermediate steps.
	static constexpr uint64_t kIntermediateStepUs = 2'500;
	static constexpr uint64_t kNativeStepUs = 10'000;
	static constexpr uint8_t kForceSyncNative = 4;
	uint64_t _last_control_step_us{0};
	uint64_t _last_native_step_us{0};
	uint8_t _intermediate_steps_since_native{0};
	Action _last_action{};
	std::array<float, rl_mc_arm_policy::kGruHiddenDim> _hidden_state{};
};
