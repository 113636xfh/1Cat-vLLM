# SM70 Worktree And PR Workflow

Use this workflow for every 1Cat SM70/V100 performance investigation, kernel
change, model-path change, benchmark, or release fix. Its purpose is to keep
one auditable integration line while allowing independent experiments to run
without overwriting source, compiler caches, or measurements.

## Integration Contract

`onecat/main` is the default integration branch. A campaign may declare a
different integration branch, but the owner must record it in the campaign
handoff before work starts. Do not commit or push directly to the integration
branch except for an explicitly authorized emergency fix.

Treat the canonical checkout as read-only coordination state. It can be dirty;
that is not permission to clean, stage, reformat, or reuse its changes. A task
is shared only after its owned branch is pushed and has a Draft PR against the
declared integration branch.

## Start A Task

Run preflight from the canonical checkout before editing:

```bash
git status --short --branch
git remote -v
git worktree list --porcelain
git fetch onecat --prune
git rev-parse onecat/main
gh auth status
```

Record the final `onecat/main` SHA as `BASE_SHA`. Check for a related open PR
before creating a duplicate path:

```bash
gh pr list --repo 1CatAI/1Cat-vLLM --state open --search "<area keywords>"
```

Create one branch and one worktree per scope. The helper resolves the canonical
checkout even when it is invoked from an existing task worktree. It defaults to
`onecat/main`; environment variables permit a declared campaign base.

```bash
tools/create-sm70-worktree.sh <short-scope>
```

The helper prints `BASE_SHA`, branch, and worktree path. Do not share any of
those three values with another active task. Use a separate worktree for
documentation-only and release work when it has a distinct review or rollback
boundary.

## Isolate Runtime State

Git isolation alone is not enough for GPU work. Pin task-owned mutable paths
before any build, benchmark, or server launch:

```bash
export TORCH_EXTENSIONS_DIR="$PWD/.cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/torchinductor"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export PROFILE_DIR="$PWD/.cache/profiles"
mkdir -p "$PROFILE_DIR"
```

Use a task-specific tmux session, API port, benchmark-result directory, and
compiler cache. Do not share a FlashQLA JIT cache, a build directory, a
profiler output directory, or a persistent server between worktrees. Pass
`$PROFILE_DIR` explicitly to each benchmark's output option. Capture the full
command and all non-default environment variables with each result.

Before GPU work, inspect GPU and Python processes. Use idle GPUs; never stop,
reset, or preempt a process unless it is proven to belong to the task. Stop
task-owned services when testing ends unless the user asked to keep them live.

## Evidence Before Promotion

Every performance PR must contain the following from the same source SHA and
workload contract:

- baseline and candidate commands, model, quantization, TP size, GPU set,
  context/input/output lengths, sampling/MTP state, backend, graph mode, and
  relevant cache settings;
- numerical and output-quality evidence appropriate to the changed path;
- focused test results and exact raw-artifact locations;
- a before/after table that separates microbenchmark projections from measured
  end-to-end prefill, TTFT, and steady decode results;
- rejected variants and known risks, including profiler perturbation or
  unavailable validation.

Do not turn a profiler percentage, a kernel-only win, or a different workload
into an endpoint speedup claim. Keep failed experiments in the worklog with a
short cause so later work does not repeat them.

## Commit, Synchronize, And Review

Make small, owned commits. Stage explicit files or hunks only; do not use
`git add -A` or stage generated build artifacts. Use signed commits:

```bash
git status --short --branch
git diff --check
git diff --stat
git add <owned paths>
git diff --cached --check
git commit -s -m "[Kernel] Describe one SM70 change"
```

Before publishing, fetch the integration branch and compare it with the task
branch. Rebase only an unpublished branch. Once published, merge the current
integration branch into the task branch; do not force-push a PR branch without
explicit coordination.

```bash
git fetch onecat --prune
git log --oneline --left-right onecat/main...HEAD
git push -u onecat <task-branch>
```

Open one Draft PR per scope using
`.github/PULL_REQUEST_TEMPLATE/sm70-performance.md`. Its base must be the
declared integration branch, normally `main`, and never `origin/main`.
The human submitter reviews every changed line and records AI assistance in the
PR. Promote it to Ready only after its documented quality and performance gates
pass. Merge accepted PRs one at a time and validate the integration branch
after hot-path changes.

## Handoff And Cleanup

Every handoff records the canonical repo, integration branch, `BASE_SHA`,
owned branch, worktree, head SHA, PR URL/state, changed files, test results,
raw artifacts, active tmux sessions, ports, GPU processes, and the next
concrete action.

After merge, verify the merge commit on `onecat/main`, stop only task-owned
processes, preserve required artifacts, and remove the worktree only after all
work is pushed and merged. For interrupted work, make and push an explicit WIP
commit; never use a shared stash as handoff.
