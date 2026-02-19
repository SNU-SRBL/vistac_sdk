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
- Status: `OPEN`
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
- Status: `OPEN`
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
