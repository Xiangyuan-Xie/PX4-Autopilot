/****************************************************************************
 *
 * Optional AM Pose RL-Tools policy adapter.
 *
 ****************************************************************************/

#pragma once

#include <cstdint>

class AmPosePolicyAdapter
{
public:
	enum class ObservationContract : uint8_t {
		Unsupported = 0,
		NoArm,
		Position,
		Velocity,
		PositionVelocity,
	};

	static constexpr int BaseObservationDim{126};
	static constexpr int Pose130ObservationDim{130};
	static constexpr int Pose134ObservationDim{134};
	static constexpr int Pose138ObservationDim{138};
	static constexpr int MaxObservationDim{Pose138ObservationDim};
	static constexpr int ActionDim{4};
	static constexpr uint8_t ArmObservationNone{0};
	static constexpr uint8_t ArmObservationPosition{1u << 0};
	static constexpr uint8_t ArmObservationVelocity{1u << 1};
	using Observation = float[MaxObservationDim];
	using Action = float[ActionDim];

	AmPosePolicyAdapter() = default;
	~AmPosePolicyAdapter();

	AmPosePolicyAdapter(const AmPosePolicyAdapter &) = delete;
	AmPosePolicyAdapter &operator=(const AmPosePolicyAdapter &) = delete;

	bool init();
	void reset();
	bool infer(const Observation &observation, Action &action);
	bool available() const { return _impl != nullptr; }
	ObservationContract observationContract() const;
	uint8_t observationDim() const { return observationDimFor(observationContract()); }
	bool usesArmPosition() const { return usesArmPosition(observationContract()); }
	bool usesArmVelocity() const { return usesArmVelocity(observationContract()); }

	static constexpr uint8_t observationDimFor(ObservationContract contract)
	{
		switch (contract) {
		case ObservationContract::NoArm:
			return Pose130ObservationDim;

		case ObservationContract::Position:
		case ObservationContract::Velocity:
			return Pose134ObservationDim;

		case ObservationContract::PositionVelocity:
			return Pose138ObservationDim;

		case ObservationContract::Unsupported:
		default:
			return 0;
		}
	}

	static constexpr uint8_t armObservationFeaturesFor(ObservationContract contract)
	{
		switch (contract) {
		case ObservationContract::Position:
			return ArmObservationPosition;

		case ObservationContract::Velocity:
			return ArmObservationVelocity;

		case ObservationContract::PositionVelocity:
			return ArmObservationPosition | ArmObservationVelocity;

		case ObservationContract::NoArm:
		case ObservationContract::Unsupported:
		default:
			return ArmObservationNone;
		}
	}

	static constexpr bool usesArmPosition(ObservationContract contract)
	{
		return (armObservationFeaturesFor(contract) & ArmObservationPosition) != 0;
	}

	static constexpr bool usesArmVelocity(ObservationContract contract)
	{
		return (armObservationFeaturesFor(contract) & ArmObservationVelocity) != 0;
	}

	static constexpr ObservationContract observationContractFor(uint8_t observation_dim, uint8_t arm_features)
	{
		if (observation_dim == Pose130ObservationDim && arm_features == ArmObservationNone) {
			return ObservationContract::NoArm;
		}

		if (observation_dim == Pose134ObservationDim && arm_features == ArmObservationPosition) {
			return ObservationContract::Position;
		}

		if (observation_dim == Pose134ObservationDim && arm_features == ArmObservationVelocity) {
			return ObservationContract::Velocity;
		}

		if (observation_dim == Pose138ObservationDim
		    && arm_features == (ArmObservationPosition | ArmObservationVelocity)) {
			return ObservationContract::PositionVelocity;
		}

		return ObservationContract::Unsupported;
	}

private:
	struct Impl;
	Impl *_impl{nullptr};
};
