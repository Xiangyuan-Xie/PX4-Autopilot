/****************************************************************************
 *
 * Common AM Position policy adapter types.
 *
 ****************************************************************************/
#pragma once

static constexpr int kPolicyObservationDim = 30;
static constexpr int kPolicyActionDim = 4;

using PolicyObservation = float[kPolicyObservationDim];
using PolicyAction = float[kPolicyActionDim];
