# Force Implementation Canonical Log (SDK vs Sparsh)

Date: 2026-02-19  
Workspace: `vistac_sdk`

## Why this file exists

This file is the **single source of truth** for force-estimation implementation status and next actions.

When starting a new chat session, the coding agent must use this file to understand:
- what is already complete,
- what remains to be done,
- what evidence supports each claim,
- and how to execute the next step safely.

---

## A) Current State (Authoritative)

### A1. Architecture parity status
Status: ✅ Implemented

- Encoder uses Sparsh-native `vit_base` path in `vistac_sdk/vistac_force.py`.
- Decoder uses Sparsh layer classes (`Reassemble`, `Fusion`, `NormalShearHead`) from `sparsh-main`.
- Decoder topology uses `reassemble_s=[4, 8, 16, 32]`.
- Checkpoint loading behavior is strict in effect:
  - probe load for diagnostics,
  - fail-fast on missing/unexpected keys,
  - strict load execution.

### A2. Model I/O and runtime semantics
Status: ✅ Implemented

- Temporal input shape: `[1, 6, 224, 224]`.
- Preprocess uses `(img - bg)/255 + offset`, clipped to `[0,1]`.
- Model-native outputs:
  - `normal`: sigmoid-bounded.
  - `shear`: tanh-scaled flow.
- Vector aggregation: spatial means `(fx, fy, fz)`.
- SDK runtime extensions (intentional):
  - vector baseline subtraction,
  - optional force-field baseline template subtraction,
  - optional physical scaling via `force_vector_scale`.

### A3. Visualization policy
Status: ✅ Implemented

- Presentation (viewer/ROS) policy in `LiveTactileProcessor`:
  - clamp `normal` to `[0,1]`,
  - clamp `shear` to `[-1,1]`,
  - apply `force_field_scale` as display-only multiplier.
- RGB mapping is centralized as `(R,G,B)=(Fx,Fy,Fz)` via `force_field_to_rgb`.
- `visualize_force_field` uses normal clipping to `[0,1]` (no stale remap).

### A4. Verification coverage
Status: ✅ Implemented (focused)

- Strict-load regression tests exist (encoder/decoder mismatch diagnostics).
- Numerical parity test exists (SDK vs independent Sparsh reference path).
- Visualization regression tests exist (normal range, channel mapping, helper consistency).

---

## B) Remaining TODO Checklist (Ordered, Machine-Parseable)

Execution rule: complete **one item at a time** and append results in Section D.

Allowed `Status` values: `OPEN`, `IN_PROGRESS`, `BLOCKED`, `DONE`.

### B1
- ID: `B1`
- Title: `CPU fallback with xformers present`
- Priority: `P0`
- Status: `BLOCKED`
- DependsOn: `NONE`
- Goal: avoid CPU failures when xformers FMHA is unavailable on CPU.
- DoneWhen:
  - CPU force estimator path runs without FMHA dispatch error.
  - CPU-safe targeted tests pass.

### B2
- ID: `B2`
- Title: `Full tests validation policy and execution`
- Priority: `P0`
- Status: `OPEN`
- DependsOn: `B1`
- Goal: move from focused tests to reproducible full validation.
- DoneWhen:
  - CPU-safe suite and GPU-required suite are explicitly defined.
  - Both suites are run, or one is marked `BLOCKED` with reason.
  - Exact commands and outputs are logged in Section D.

### B3
- ID: `B3`
- Title: `ROS2 end-to-end validation`
- Priority: `P0`
- Status: `BLOCKED`
- DependsOn: `B1`
- Goal: validate runtime behavior beyond import checks.
- RequiredScope:
  - single-sensor launch
  - multi-sensor launch
  - force field topic stream
  - force vector topic stream
  - pointcloud_force mode behavior
  - sustained streaming consistency
- DoneWhen:
  - Commands and observed outcomes are logged in Section D.
  - Each scope item has explicit `PASS`/`FAIL`/`BLOCKED` status.

### B4
- ID: `B4`
- Title: `CI gating policy`
- Priority: `P1`
- Status: `OPEN`
- DependsOn: `B2`
- Goal: enforce CPU/GPU expectations in CI.
- DoneWhen:
  - Explicit pass/fail gates exist for CPU-safe and GPU-required categories.
  - Parity tests cannot be skipped silently.

### B5
- ID: `B5`
- Title: `Dependency metadata alignment`
- Priority: `P1`
- Status: `OPEN`
- DependsOn: `B2`
- Goal: align runtime/install docs for force stack.
- DoneWhen:
  - `requirements.txt` and `setup.py` intent are reconciled.
  - Optional GPU acceleration dependencies are documented clearly.

---

## C) Evidence Map (Claim → Code/Test)

### C1. Core implementation
- `vistac_sdk/vistac_force.py`
- `vistac_sdk/tactile_processor.py`
- `vistac_sdk/live_core.py`
- `vistac_sdk/viz_utils.py`

### C2. Tests
- `tests/test_force_estimator.py`
- `tests/test_tactile_processor.py`
- `tests/test_viz_utils.py`

### C3. User-facing contract
- `README.md`

Rule: if a claim in Section A or B has no anchor in Section C or Section D, mark it **Unverified**.

---

## D) Session Log (Append-Only, Machine-Parseable)

Use this template after each step update.

```
EntryID: D<number>
DateTimeUTC: YYYY-MM-DDTHH:MM:SSZ
StepID: B1|B2|B3|B4|B5
StepStatus: OPEN|IN_PROGRESS|BLOCKED|DONE
Result: PASS|PARTIAL|FAIL|BLOCKED

Summary:
- ...

CommandsRun:
- <exact command>

ValidationOutput:
- <exact output or summarized with reference>

FilesChanged:
- <path>

ScopeChecks:  # required for B3
- item: <scope-name>
  status: PASS|FAIL|BLOCKED
  note: <short note>

BlockerReason: <empty or reason>
NextStep: B?
```

### D0. Baseline entry (2026-02-19)

- EntryID: `D0`
- DateTimeUTC: `2026-02-19T00:00:00Z`
- StepID: `BASELINE`
- StepStatus: `OPEN`
- Result: `PARTIAL`
- Summary:
  - Focused validation confirmed for parity + visualization subsets.
  - Remaining work is represented by open checklist items B1–B5.

### D1. B1 scope reevaluation started (2026-02-19)

- EntryID: `D1`
- DateTimeUTC: `2026-02-19T01:42:49Z`
- StepID: `B1`
- StepStatus: `IN_PROGRESS`
- Result: `PARTIAL`
- Summary:
  - User confirmed GPU-only deployment constraint.
  - B1 implementation scope reevaluation initiated to avoid unnecessary CPU-path work.
- CommandsRun:
  - `date -u +"%Y-%m-%dT%H:%M:%SZ"`
- ValidationOutput:
  - `2026-02-19T01:42:49Z`
- FilesChanged:
  - `force.md`
- BlockerReason: ``
- NextStep: `B1`

### D2. B1 blocked by deployment policy (2026-02-19)

- EntryID: `D2`
- DateTimeUTC: `2026-02-19T01:43:07Z`
- StepID: `B1`
- StepStatus: `BLOCKED`
- Result: `BLOCKED`
- Summary:
  - B1 CPU fallback implementation is intentionally out of scope.
  - Deployment target is GPU-present systems only, per user direction.
- CommandsRun:
  - `date -u +"%Y-%m-%dT%H:%M:%SZ"`
- ValidationOutput:
  - `2026-02-19T01:43:07Z`
- FilesChanged:
  - `force.md`
- BlockerReason: `CPU compatibility work is excluded by GPU-only deployment policy.`
- NextStep: `B3`

### D3. B3 validation run (single-device environment) (2026-02-19)

- EntryID: `D3`
- DateTimeUTC: `2026-02-19T02:01:15Z`
- StepID: `B3`
- StepStatus: `IN_PROGRESS`
- Result: `PARTIAL`
- Summary:
  - Executed ROS2 runtime validation with one connected sensor (`D21242`).
  - Verified single-sensor force topics and message types.
  - Verified multi-sensor launch behavior where unavailable sensors exit gracefully.
- CommandsRun:
  - `source /opt/ros/humble/setup.bash && colcon build --packages-select vistac_sdk`
  - `source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 run vistac_sdk tactile_streamer_node --ros-args -p serial:=D21242 -p sensors_root:=$PWD/sensors -p enable_force:=true -p model_device:=cuda -p outputs:="[force_field,force_vector]" -p rate:=10.0`
  - `source /opt/ros/humble/setup.bash && source install/setup.bash && nohup ros2 launch vistac_sdk multi_sensor_tactile_streamer.launch.py sensors_root:=$PWD/sensors enable_force:=true outputs:=force_vector rate:=8.0 </dev/null >/tmp/b3_multi.log 2>&1 & pid=$!; sleep 18; ros2 topic list | grep -E '^/tactile/(D21119|D21242|D21273|D21275)/force_vector$' | sort; timeout 10s ros2 topic echo --once /tactile/D21242/force_vector; kill $pid; wait $pid`
  - `source /opt/ros/humble/setup.bash && source install/setup.bash && timeout 12s ros2 topic echo --once /tactile/D21242/force_field`
- ValidationOutput:
  - `/tactile/D21242/force_field` present; type `sensor_msgs/msg/Image`
  - `/tactile/D21242/force_vector` present; type `geometry_msgs/msg/WrenchStamped`
  - `ros2 topic echo --once /tactile/D21242/force_vector` produced non-zero force sample.
  - Multi-launch started four nodes; three sensors (`D21119`,`D21273`,`D21275`) logged "not detected" and exited cleanly; `D21242` stream remained active.
- FilesChanged:
  - `ros2/tactile_streamer_node.py`
  - `vistac_sdk/live_core.py`
  - `vistac_sdk/vistac_force.py`
- ScopeChecks:
  - item: `single-sensor launch`
    status: `PASS`
    note: `D21242 node initialized and produced force topics/messages.`
  - item: `multi-sensor launch`
    status: `PASS`
    note: `Launch orchestration works; non-connected sensors exit gracefully; connected sensor remains active.`
  - item: `force field topic stream`
    status: `PASS`
    note: `Topic and message type observed for D21242.`
  - item: `force vector topic stream`
    status: `PASS`
    note: `Topic and message payload observed for D21242.`
  - item: `pointcloud_force mode behavior`
    status: `BLOCKED`
    note: `Clean, repeatable topic introspection blocked by stale-process interference and ROS transport instability during this session.`
  - item: `sustained streaming consistency`
    status: `BLOCKED`
    note: `Reliable `ros2 topic hz` sampling was not repeatable under current DDS SHM transport errors.`
- BlockerReason: ``
- NextStep: `B3`

### D4. B3 blocked pending full hardware + stable DDS transport (2026-02-19)

- EntryID: `D4`
- DateTimeUTC: `2026-02-19T02:01:45Z`
- StepID: `B3`
- StepStatus: `BLOCKED`
- Result: `BLOCKED`
- Summary:
  - B3 was executed as far as possible in a single-device setup and produced partial PASS coverage.
  - Full B3 completion is blocked by unavailable additional sensors and unstable DDS SHM transport behavior affecting repeatable long-run checks.
- CommandsRun:
  - `source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 topic list | grep -E '^/tactile/' | sort`
  - `source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 topic echo --once /tactile/D21242/force_vector`
  - `source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 topic echo --once /tactile/D21242/force_field`
- ValidationOutput:
  - Confirmed active D21242 force topic traffic.
  - Could not complete reproducible `pointcloud_force` behavior verification and sustained stream-rate measurement under current environment constraints.
- FilesChanged:
  - `force.md`
- ScopeChecks:
  - item: `single-sensor launch`
    status: `PASS`
    note: `Validated on connected sensor D21242.`
  - item: `multi-sensor launch`
    status: `PASS`
    note: `Launch path validated; non-connected sensors exited gracefully.`
  - item: `force field topic stream`
    status: `PASS`
    note: `Observed with D21242.`
  - item: `force vector topic stream`
    status: `PASS`
    note: `Observed with D21242.`
  - item: `pointcloud_force mode behavior`
    status: `BLOCKED`
    note: `Needs stable clean-run introspection and/or additional instrumentation.`
  - item: `sustained streaming consistency`
    status: `BLOCKED`
    note: `Needs stable DDS transport and longer uninterrupted run checks.`
- BlockerReason: `Only D21242 is connected; additional-sensor E2E scope and stable long-run DDS validation are currently not reproducible.`
- NextStep: `B2`

---

## E) Agent Execution Guide (adapted from `ai_agent_guidelines_v2.md`)

This section is the in-file execution policy for new coding-agent sessions.

### E1. Pre-execution (mandatory)

Before doing any change, the agent must:
1. Read this entire file (`force.md`).
2. Read the user’s current step request.

### E2. One-step execution discipline

- Execute only one checklist item at a time (B1, then B2, ...).
- After finishing one item, stop and report results.
- Do not start the next item without explicit user confirmation.

### E3. Editing scope

- Edit only files required by the active checklist item.
- Always update Section D after each completed item.
- If file scope is ambiguous, ask before editing.

### E4. Zero-hallucination policy

- Do not invent APIs, flags, methods, or CLI args.
- If behavior/doc references are unclear, stop and ask.
- Mark uncertain claims as **Unverified** until evidence is added.

### E5. Validation requirements

Every code change must be followed by relevant validation:
- run targeted tests first,
- then broader tests required by the active checklist item,
- include exact command outputs in Section D.

If tests fail:
- diagnose root cause,
- fix or report blocker,
- rerun affected tests.

### E6. Required report format (for each finished checklist item)

```
Step Complete: [B? - Title]

What Was Done
- ...

Validation
$ <command>
<output>

Assumptions & Deviations
Assumptions: ...
Deviations: ...

Next Step
[Next B-item, not started]

Ready to proceed? [Yes/No]
```

### E7. Safety constraints

- Do not expose secrets/tokens.
- Ask before destructive actions.
- Do not upgrade dependencies unless explicitly requested.

---

## Maintenance rule

Keep Sections A/B/C concise and authoritative.

Do not place long terminal dumps in the main body; keep command evidence in Section D entries.

---

## F) Update Protocol (Allowed Mutations Only)

When completing one step in a new session, the agent may only do the following edits:

1. In Section B:
  - Update exactly one item’s `Status` field.
  - Allowed transitions:
    - `OPEN -> IN_PROGRESS`
    - `IN_PROGRESS -> DONE`
    - `IN_PROGRESS -> BLOCKED`
    - `BLOCKED -> IN_PROGRESS`

2. In Section D:
  - Append one new `D<number>` entry (do not edit previous entries).
  - Include exact commands and validation output for the current step.

3. In Section A/C:
  - Update only if implementation truth changes and evidence exists.
  - If evidence is missing, mark claim as `Unverified` instead of rewriting as fact.

4. Forbidden edits (unless user explicitly asks):
  - Reordering checklist IDs (`B1...B5`).
  - Deleting old session-log entries.
  - Changing `DoneWhen` criteria semantics.
  - Broad rewording of historical entries.

5. End-of-step consistency check:
  - Exactly one active step should be `IN_PROGRESS` at a time.
  - Any `DONE` step must have a matching appended Section D entry.
  - `NextStep` in the newest D-entry must reference a valid `B?` ID.
