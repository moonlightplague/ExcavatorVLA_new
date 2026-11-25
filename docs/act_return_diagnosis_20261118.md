# ACT return and cross-cycle control diagnosis

The current short-window ACT can predict expert return motions on recorded
validation observations, but the source publication contains unlabelled bucket
command-target resets. This is the same class of control incompatibility that
the earlier PI05 relabelling experiment addressed. A small paired physical
probe supports correcting this incompatibility before increasing return
sampling or adding history. It does not establish the cause of every model
failure or demonstrate that a retrained model will succeed.

## Frozen-model diagnostic

Checkpoint: `act_shortchunk_repeat40k_20261117_01`, step 40k, predict 10 /
execute 3. All 200 source parents and the original 140/20/40 assignments remain
unchanged. The diagnostic selects 941 distinct observations from the 20
validation parents, up to eight observations per parent per group. Groups may
overlap; these are stratified diagnostics, not full validation scores. Stage
and return labels select offline groups and never enter model inputs.

| Validation group | Samples | First-action MAE (rad/s) | Moving direction accuracy |
|---|---:|---:|---:|
| Return start, first ten recorded return frames | 160 | 0.04373 | 99.08% |
| Active return | 160 | 0.05541 | 97.63% |
| Quiet return | 160 | 0.00343 | Not applicable |
| Unload | 160 | 0.01365 | 100.00% |
| First-cycle cut | 160 | 0.15183 | 84.50% |
| Later-cycle cut | 160 | 0.15100 | 84.44% |

At return start, none of the samples with expert swing magnitude above
0.2 rad/s is predicted within 0.02 rad/s of zero. Active return has one such
near-zero prediction out of 105 active-swing samples. Training return frames
comprise 10.82% of the training set; 87.12% of these frames have at least one
command magnitude above 0.05 rad/s. This does not support a blanket claim that
the policy has failed to learn return actions or that return examples are
mostly idle. It does not test recovery from off-trajectory images/states.

## Command-target audit

For each adjacent pair within an original parent, reconstruct the command
target as measured joint position plus recorded tracking error. Compare ten
times its adjacent difference with the recorded action. Never difference
across parent boundaries or interpolate missing frames.

| Split | Adjacent transitions | Transitions with any error >0.1 rad/s |
|---|---:|---:|
| Train | 345,041 | 2,953 (0.856%) |
| Validation | 49,495 | 409 (0.826%) |
| Test, diagnostic only | 98,720 | 834 (0.845%) |

The large discrepancies are in bucket commands during cut/unload, not the
return-tagged segment or scoop-index boundary. For example, export 16 frame
186 has a zero bucket action while the recorded bucket target changes from
45 degrees to approximately -11.69 degrees. Reproducing that target endpoint
in 0.1 seconds requires about -9.90 rad/s. The original profile allows only
6 rad/s bucket command-target slew. Measured mechanical speed is a different
quantity. Training alone cannot make the original labels reproduce this reset.

## Paired physical probe

Use training parent 0, its first complete recorded cycle, 233 actions, identical
scene/seed/backend, and expert actions only. Compare original labels under the
original bucket-six profile with reconstructed labels under the explicit
experimental `excavator_bucket_slew20_v1` profile already used by PI05. The
reconstruction changes four action frames and causal previous-action state
fields; observed joints, images, geometry and source files remain untouched.

| Metric | Original controls | Reconstructed controls |
|---|---:|---:|
| Command-target maximum error | 1.41749 rad | 0.000379 rad |
| Measured-joint maximum error | 1.42075 rad | 0.07650 rad |
| Final bucket joint error | 80.97 degrees | -0.004 degrees |
| Final delivered source particles | 15 | 40 |

These results compare two control-label/profile combinations and are not an
isolated action-limit ablation or a learned-model success result. The physical
probe supports the control incompatibility diagnosis; it does not prove that
all multi-cycle failures have this cause.

## Next gate

Two reconstructed expert-only ten-cycle replays were launched on training
parent 0 and validation parent 22, each with a separate output directory.
They check command-target parity, measured-joint parity and per-cycle net
delivered mass. No learned policy is evaluated, and recorded task goals are
not used as learned-policy inputs. Three distinct complete ten-cycle replays
and a separately versioned dataset publication remain required before any
formal training on revised labels. No new training has started.

History `[-6,-3,0]` and targeted sampling remain candidate follow-ups. They
should be evaluated after the executable control labels are verified.

Artifacts:

- `outputs/experiments/act_return_diagnosis_20261118_01/`
- `outputs/experiments/act_control_replay_probe_20261118_01/`
- `outputs/experiments/act_control_replay10_ep0000_20261118_01/`
- `outputs/experiments/act_control_replay10_ep0022_20261118_01/`

Checks: three control-audit regressions and the ten existing action-relabel
regressions passed. The existing pytest package is unavailable in the model
environment; these unittest suites ran directly without environment changes.
