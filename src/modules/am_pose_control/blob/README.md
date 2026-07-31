# AM Pose Deployment Blob

Deploy one matching pair in this directory:

- `policy.h`: RL-Tools model weights and executable graph.
- `policy.json`: a compact deployment manifest, not the full ACELab export metadata.

The manifest must remain below 4 KiB and declare the observation layout used by
the model:

```json
{
  "manifest_version": 1,
  "policy_name": "checkpoint_name",
  "observation_dim": 134,
  "arm_observation": "position",
  "action_dim": 4,
  "action_semantics": "normalized_throttle"
}
```

Allowed `arm_observation` values are `none` (130D), `position` (134D),
`velocity` (134D), and `position_velocity` (138D). The build rejects a
manifest whose dimension does not match its declared layout.

Keep the full ACELab export JSON outside this directory. It is useful for
training-side reproduction, but PX4 only needs the compact manifest above.
