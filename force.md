# Force Implementation Review (SDK vs Sparsh)

Date: 2026-02-13  
Workspace: `vistac_sdk`

## Executive Summary

This report reviews the current SDK force implementation against the original Sparsh codebase, covering:
- End-to-end data flow
- Input/output shapes and value ranges
- Force-field and force-vector semantics
- Visualization behavior in app and ROS2

### Top conclusions

1. **Pipeline intent is aligned**: temporal pair input, background differencing, force-field inference, and vector aggregation are conceptually consistent with Sparsh.
2. **Model implementation fidelity is not exact**: the SDK uses a custom encoder/decoder implementation and `strict=False` weight loading; this is a meaningful deviation from Sparsh reference modules.
3. **Force-field visualization behavior differs by design**: SDK live layer clips shear before RGB mapping.
4. **Potential visualization bug identified**: `visualize_force_field()` remaps `normal` as if it were in `[-1, 1]`, while live layer already canonicalizes `normal` to `[0, 1]`.

## Agent Quick Start (5 Rules)

- Read this report fully before editing code; prioritize **Action Items Checklist** top-to-bottom.
- Treat the live-layer shear clamp `[-1, 1]` for force-field RGB visualization as an **intentional requirement**.
- Run focused tests after each change and include full output in status updates.
- Do not claim Sparsh parity unless architecture and checkpoint loading are verified with strict checks/parity tests.
- If any requirement is ambiguous, ask the user before implementing assumptions.

## Action Items Checklist

- [x] Validate and decide final target: strict Sparsh parity vs Sparsh-inspired runtime integration.
- [x] Replace custom encoder/decoder with native Sparsh modules (or prove exact architectural equivalence).
- [x] Switch checkpoint loading to strict validation and record missing/unexpected keys if any.
- [x] Add parity test harness: same frame pair through Sparsh reference and SDK, compare `normal`, `shear`, and aggregated vector.
- [x] Keep live-layer shear clamp `[-1, 1]` for RGB visualization as intentional product decision.
- [x] Fix `visualize_force_field()` normal remap path so `[0,1]` inputs are not remapped as `[-1,1]`.
- [x] Add regression tests for force-field visualization ranges and channel mapping.
- [x] Add documentation note in README/API for force-field visualization policy and non-physical display scaling.
- [x] Re-run full tests and add a short validation log for hardware and ROS2 streaming checks.

---

## Scope and References

### SDK files reviewed
- `vistac_sdk/vistac_force.py`
- `vistac_sdk/tactile_processor.py`
- `vistac_sdk/live_core.py`
- `vistac_sdk/viz_utils.py`
- `apps/live_viewer.py`
- `ros2/tactile_streamer_node.py`
- `tests/test_force_estimator.py`
- `tests/test_viz_utils.py`

### Sparsh files reviewed
- `sparsh-main/tactile_ssl/downstream_task/forcefield_sl.py`
- `sparsh-main/tactile_ssl/downstream_task/force_sl.py`
- `sparsh-main/tactile_ssl/downstream_task/utils_forcefield/layers/Head.py`
- `sparsh-main/tactile_ssl/downstream_task/utils_forcefield/layers/Reassemble.py`
- `sparsh-main/tactile_ssl/data/digit/utils.py`
- `sparsh-main/tactile_ssl/data/vision_based_interactive.py`
- `sparsh-main/tactile_ssl/test/demo_t1_forcefield.py`
- `sparsh-main/config/task/digit_forcefield.yaml`
- `sparsh-main/config/task/t1_force_estimation.yaml`
- `sparsh-main/config/data/digit.yaml`

---

## 1) End-to-End Data Flow Comparison

## 1.1 Sparsh reference flow
1. Capture tactile frames (interactive demo / dataset loader).
2. Background differencing (`compute_diff`):
   - `diff = (img - bg) / 255 + offset`
   - clip to `[0, 1]`
3. Resize + `ToTensor` (no ImageNet normalization).
4. Temporal concat to 6 channels (`[3(t), 3(t-n)]`).
5. Encoder + forcefield decoder forward pass.
6. Outputs:
   - `normal` (sigmoid)
   - `shear` (tanh * `scale_flow=20`)
7. Optional aggregation to vector (mean over spatial dimensions).

## 1.2 SDK current flow
1. Camera frame acquisition in `LiveTactileProcessor`.
2. Runtime background collection (10-frame average) and load into estimators.
3. Temporal buffering in `ForceEstimator`.
4. Preprocess with same differencing formula, RGB conversion, resize, tensor conversion, temporal concat.
5. Force inference via SDK encoder/decoder implementation.
6. `TactileProcessor` returns force outputs (`None` during warmup).
7. `LiveTactileProcessor` applies presentation canonicalization/scaling.
8. Viewer and ROS2 publish/render force data.

## 1.3 Flow alignment verdict
- **Aligned conceptually**.
- **Not fully equivalent implementation-wise** due to custom model stack and output presentation policy in live layer.

---

## 2) Input/Output Shapes and Range Semantics

## 2.1 Inputs

### Sparsh
- Input tensor: `[B, 6, 224, 224]`
- Channels are two RGB frames concatenated.

### SDK
- Input tensor: `[1, 6, 224, 224]` (per inference call)
- Same concatenation concept and size.

**Status**: ✅ Shape/format intent matches.

## 2.2 Force-field outputs

### Sparsh
- `normal`: sigmoid output, expected `[0, 1]`
- `shear`: tanh output scaled by `scale_flow=20`, expected roughly `[-20, 20]`

### SDK
- Decoder applies:
  - `normal = sigmoid(...)`
  - `shear = tanh(...) * 20`
- Estimator returns these as model-native arrays.

**Status**: ✅ Head activation/range semantics match at estimator level.

## 2.3 Force-vector outputs

### Sparsh forcefield path
- Aggregates by spatial mean:
  - `fx = mean(shear_x)`
  - `fy = mean(shear_y)`
  - `fz = mean(normal)`

### SDK
- Same aggregation formulas (mean over 224×224).
- Adds:
  - runtime baseline subtraction for vector
  - `force_vector_physical` scaling by sensor config (`force_vector_scale`)

**Status**: ✅ Aggregation math matches; ➕ SDK adds practical runtime/physical scaling extensions.

---

## 3) Differences from Original Sparsh

## 3.1 Model implementation fidelity (critical)

### Findings
- Sparsh task config uses official model modules (`vit_base` + `ForceFieldDecoder`).
- SDK uses custom re-implemented encoder/decoder classes.
- SDK loads weights with `strict=False` for both encoder and decoder.

### Why this matters
`strict=False` can hide key mismatches and structural deviations. Even if outputs look plausible, this can diverge from official Sparsh behavior and calibration characteristics.

### Risk level
**High** for claiming strict equivalence to Sparsh reference inference.

---

## 3.2 Decoder reassemble scale path differs

### Sparsh reference
- Reassemble scales constrained to `[4, 8, 16, 32]`.

### SDK current
- Uses `[4, 2, 1, 2]` in the custom decoder implementation.

### Impact
Multi-scale feature fusion behavior differs from reference architecture. This is a structural difference, not just post-processing.

### Risk level
**High** (core model path).

---

## 3.3 Live-layer force-field clipping/scaling policy

### SDK behavior
In `LiveTactileProcessor.get_latest_output()`:
- `normal` is clamped to `[0, 1]`
- `shear` is clamped to `[-1, 1]`
- optional `force_field_scale` is applied for presentation

### Sparsh demo behavior
- Uses `rad_max=20` normalization strategy for shear visualization (not hard clipping to `[-1, 1]` in model units).

### Impact
SDK intentionally compresses/limits displayed shear dynamic range for visualization consistency.

### Decision note (user-provided)
**For force field visualization, clamping shear to `[-1, 1]` in the SDK live layer is an intentional user decision for RGB visualization** (as requested).

---

## 3.4 Force vector task coverage difference

### Sparsh
Has two relevant paradigms:
1. force-field prediction (`digit_forcefield`)
2. direct force-vector probe (`t1_force_estimation` via `ForceLinearProbe`)

### SDK
Implements vector from force-field aggregation (plus baseline/scaling), not the direct vector probe branch.

### Impact
Behavior corresponds to Sparsh **forcefield** path, not the separate Sparsh **direct vector probe** path.

---

## 4) Visualization Review (App, Utils, ROS2)

## 4.1 RGB channel mapping consistency

### SDK mapping used
- `R = Fx`
- `G = Fy`
- `B = Fz`

This mapping appears consistent in live viewer and ROS2 force image publication.

**Status**: ✅ Consistent mapping in SDK components.

## 4.2 Potential bug in `visualize_force_field()`

Current utility does:
- `normal_norm = (normal + 1) / 2`

But live layer already canonicalizes `normal` to `[0,1]`. Applying `(x+1)/2` again maps `[0,1] -> [0.5,1]`, reducing visual contrast and biasing brightness.

### Impact
Displayed normal-force intensity may be skewed upward.

### Severity
**Medium** (visual correctness issue).

---

## 5) Force Baseline and Physical Scaling

## 5.1 SDK extensions
- `force_vector_baseline` computed from background pair and subtracted at runtime.
- Optional per-pixel `force_field` baseline subtraction.
- `force_vector_physical` computed with `force_vector_scale` from YAML.

## 5.2 Sparsh reference relation
These are practical SDK additions and not part of core original Sparsh forcefield inference contract.

**Status**: Intentional extension; not a mismatch bug by itself.

---

## 6) Tests and Validation Coverage

### Current SDK tests include
- Decoder output ranges/shapes
- Warmup behavior (`None` until temporal buffer ready)
- Baseline subtraction behavior
- Visualization utility behavior

### Gaps
- No strict model-equivalence test against Sparsh reference outputs on a shared input batch.
- No explicit regression test for normal-channel remapping issue in `visualize_force_field()`.

---

## 7) Priority Findings

## P0 (highest)
1. **Architecture fidelity risk**: custom encoder/decoder + `strict=False` loading.
2. **Reassemble scale divergence** from Sparsh reference architecture.

## P1
3. **Visualization-range policy difference** (intentional): live shear clamp to `[-1,1]`.
4. **Potential normal remap bug** in `visualize_force_field()`.

## P2
5. Force-vector branch differs from Sparsh direct vector probe task (design choice).

---

## 8) Recommended Next Actions

1. **Lock down model fidelity**
   - Use native Sparsh modules for inference (or exact architecture clone) and validate with `strict=True` loading.
2. **Add parity test harness**
   - Run same frame pair through Sparsh reference and SDK; compare normal/shear distributions and vector aggregates.
3. **Fix visualization utility normal mapping**
   - Treat `normal` as `[0,1]` in utility when called from live-canonicalized outputs.
4. **Document visualization policy clearly**
   - Keep your intentional shear clamp decision explicit in README/API docs.

---

## Final Note

The SDK implementation is functional and thoughtfully engineered for runtime use, but it should be described as a **Sparsh-inspired / Sparsh-weight-based integration** unless architecture-level parity is formally validated. The live-layer shear clamp for RGB visualization is intentionally preserved per your decision.

---

## 9) Strict Parity Decision Record (Approved)

### Selected target
**Strict Sparsh parity for SDK force estimation (`vistac_force`)**.

### Rationale
- Sparsh-native encoder strict-load probe passes (`missing=0`, `unexpected=0`, `strict=True`).
- Sparsh-native decoder topology strict-load probe passes (`missing=0`, `unexpected=0`, `strict=True`).
- Current SDK deviations are implementation choices, not model-checkpoint constraints.

### Implementation policy
1. Build encoder using Sparsh-native `vit_base` configuration equivalent to checkpoint training setup for forcefield inference.
2. Build decoder using Sparsh-native forcefield decoder topology (`Reassemble`, `Fusion`, `NormalShearHead`) with `reassemble_s=[4,8,16,32]`.
3. Use **strict checkpoint loading** for both encoder and decoder and fail fast on any missing/unexpected key.
4. Keep live-layer shear clamp `[-1,1]` for RGB visualization as a presentation-only policy outside estimator model-native outputs.

### Acceptance criteria (must all pass)
- **A1: Architecture parity**
   - `vistac_force` uses Sparsh-native encoder/decoder modules (or exact source-equivalent classes) for inference path.
- **A2: Checkpoint strictness**
   - Encoder load: `missing_keys == 0`, `unexpected_keys == 0`, `strict=True` pass.
   - Decoder load: `missing_keys == 0`, `unexpected_keys == 0`, `strict=True` pass.
- **A3: Numerical parity harness**
   - Same temporal frame pair through Sparsh reference path and SDK path; compare:
      - `normal`: mean absolute error and max absolute error within configured tolerance.
      - `shear`: per-channel mean/max absolute error within configured tolerance.
      - aggregated vector `(fx, fy, fz)`: absolute error per axis within configured tolerance.
- **A4: Separation of concerns**
   - `ForceEstimator` returns model-native outputs; visualization clipping/scaling remains in live/viewer/ROS presentation layers.
- **A5: Regression coverage**
   - Add tests that fail on non-strict loads and on decoder topology drift from Sparsh reference config.

## 10) Validation Log (Current Workspace)

### Full test rerun
- `runTests` (workspace-wide): **passed=66, failed=0**.

### Hardware streaming sanity check
- Command: short `LiveTactileProcessor` force run on sensor `D21242` with `enable_force=True`.
- Result: **PASS** (5 valid force frames observed).
- Observed ranges (sample):
   - `normal[min,max]` around `0.0585..0.0620`
   - `shear_abs_p99` around `0.237..0.250`
   - non-zero `force_vector` values produced per frame.

### ROS2 streaming readiness check
- `ros2 --help`: **PASS** (ROS2 CLI available).
- Python import checks:
   - `import rclpy`: **FAIL** (`ModuleNotFoundError: No module named 'rclpy'`)
   - `import ros2.tactile_streamer_node`: **FAIL** (blocked by missing `rclpy`)
- Status: ROS2 runtime streaming could not be executed in this Python environment until `rclpy` is installed/available.

---

## Appendix A) AI Agent Guideline (Copied from PLAN.md)

## Implementation Guidelines for AI Coding Agents

**CRITICAL**: These guidelines MUST be followed during implementation.

### 0. Agent Quick Start ⚠️

- Read this report fully before editing code; prioritize **Action Items Checklist** top-to-bottom.
- Treat the live-layer shear clamp `[-1, 1]` for force-field RGB visualization as an **intentional requirement**.
- Run focused tests after each change and include full output in status updates.
- Do not claim Sparsh parity unless architecture and checkpoint loading are verified with strict checks/parity tests.
- If any requirement is ambiguous, ask the user before implementing assumptions.

### 1. One Step at a Time ⚠️

- **Execute ONLY ONE step** from the Implementation Steps section per iteration
- After completing a step:
   1. Run all relevant tests
   2. Debug any errors completely
   3. Show results to user with evidence (test output, file contents, screenshots)
   4. Wait for user confirmation before proceeding to next step
- **NEVER** skip ahead or combine steps
- **NEVER** assume a step works without testing

### 2. Read and Update Plan ⚠️

**Before starting ANY work**:
1. Read PLAN.md completely, line by line
2. Identify which step you're implementing
3. Note all requirements, specifications, and constraints for that step

**After completing each step**:
1. Update this plan with:
    - ✅ Mark step as complete
    - Add any deviations from original spec
    - Document any issues encountered and resolutions
    - Note any assumptions made (see below)
2. Commit updated PLAN.md with descriptive message

### 3. No Assumptions - Always Ask ⚠️

**STOP and ASK the user if**:
- Any specification is unclear or ambiguous
- You need to choose between multiple valid approaches
- You encounter unexpected behavior
- External documentation conflicts with plan
- A dependency version is not specified
- File structure differs from plan
- You need to make ANY assumption about:
   - File paths
   - Function signatures
   - Data formats
   - Configuration values
   - User preferences

**Always look for official documentation**:
- Check official GitHub repos (not blog posts)
- Read HuggingFace model cards directly
- Verify API documentation from source
- Link to official docs in your questions to user

### 4. Brief User After Each Step ⚠️

**Required template for step completion**:

```markdown
## Step [N] Complete: [Step Name]

### What Was Done
- [Bullet list of concrete actions taken]
- [Files created/modified]
- [Commands run]

### Test Results
[Paste actual test output, not "tests passed"]
```bash
$ command_run
output here
```

### Assumptions Made
- [List ANY assumptions, or state "None"]
- [If assumptions made, explain why and ask for confirmation]

### Deviations from Plan
- [List any deviations, or state "None"]
- [Explain rationale for deviations]

### Files Changed
- [file1.py] - [what changed]
- [file2.py] - [what changed]

### Next Step
[State what the next step will be, but DON'T start it yet]

**Ready to proceed? [Yes/No]**
```

**Wait for user response before continuing**

### 5. Testing Requirements ⚠️

For EACH step:
1. Write tests BEFORE implementation (if applicable)
2. Run tests AFTER implementation
3. Show FULL test output (not summaries)
4. Debug failures completely before moving on
5. Document test commands in brief

### 6. Error Handling ⚠️

When errors occur:
1. **DO NOT** skip over errors
2. Read error messages completely
3. Check logs/stack traces
4. Try obvious fixes (typos, imports, paths)
5. If not obvious → ASK user, don't guess
6. Document error and resolution in brief

### 7. Version Control ⚠️

After EACH step:
```bash
git add [files]
git commit -m "Step [N]: [concise description]

- [what was implemented]
- [tests status]
- [any issues resolved]"
```

### 8. Code Quality ⚠️

- Follow existing code style in repository
- Add docstrings to all functions/classes
- Include type hints where applicable
- Comment complex logic
- Keep functions focused and small

### 9. Validation Checklist ⚠️

Before marking step complete:
- [ ] Code runs without errors
- [ ] Tests pass (or explain why no tests yet)
- [ ] Follows plan specifications exactly
- [ ] No assumptions made, or assumptions documented and approved
- [ ] User briefed with template above
- [ ] Changes committed to git
- [ ] PLAN.md updated
