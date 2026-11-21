# Launch-Only Tutorial Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep the migration fixes on a dedicated branch and make the tutorial initiate batch jobs without waiting.

**Architecture:** Create branch `tutorial-launch-comparison`, replace the sole blocking SCF launch, and clarify staged launch dependencies in the notebook. Preserve existing downstream `start()` calls and inspect/read behavior.

**Tech Stack:** Git, VS Code Jupyter notebook, openEO Python client, pytest.

## Global Constraints

- Launch cells must call `start()` and must not call `start_and_wait()` or polling/wait helpers.
- Downstream jobs remain staged until their upstream job is finished.
- No source or test behavior changes in this migration task.

## Tasks

### Task 1: Branch migration

- [x] Create `tutorial-launch-comparison` from the current branch while preserving uncommitted work.
- [ ] Commit the existing migration changes plus launch-only tutorial edits.

### Task 2: Launch-only tutorial

- [x] Replace the SCF `start_and_wait()` call with `start()` and display the job object.
- [x] Update the tutorial intro to document launch-only and staged upstream dependencies.
- [ ] Verify no blocking `start_and_wait()` call remains.

### Task 3: Verification

- [ ] Run the existing pytest task.
- [ ] Review `git diff --stat` and `git diff --stat` after commit.
