#!/usr/bin/env bash
# Create an isolated SM70 task worktree from the declared 1Cat integration ref.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: tools/create-sm70-worktree.sh <scope> [--base <remote/branch>] [--dry-run]

Create a unique agent/sm70-* branch and sibling worktree. By default the base
is onecat/main. Override the remote and branch with VLLM_PUSH_REMOTE and
VLLM_INTEGRATION_BRANCH, or pass --base for a declared campaign branch.

Optional environment variables:
  VLLM_WORKTREE_ROOT    Directory for new worktrees.
  VLLM_WORKTREE_OWNER   Branch owner prefix (default: agent).
  VLLM_PUSH_REMOTE      Push remote (default: onecat).
  VLLM_INTEGRATION_BRANCH  Integration branch (default: main).
EOF
}

scope=""
base=""
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            base="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        -*)
            usage >&2
            exit 2
            ;;
        *)
            [[ -z "$scope" ]] || { usage >&2; exit 2; }
            scope="$1"
            shift
            ;;
    esac
done

[[ -n "$scope" ]] || { usage >&2; exit 2; }
[[ "$scope" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || {
    echo "scope must use lowercase letters, digits, dots, underscores, or hyphens" >&2
    exit 2
}

repo_root="$(git rev-parse --show-toplevel)"
common_git_dir="$(git rev-parse --git-common-dir)"
if [[ "$common_git_dir" != /* ]]; then
    common_git_dir="${repo_root}/${common_git_dir}"
fi
common_git_dir="$(cd "$common_git_dir" && pwd)"
if [[ "$(basename "$common_git_dir")" == ".git" ]]; then
    canonical_root="$(dirname "$common_git_dir")"
else
    canonical_root="$repo_root"
fi
push_remote="${VLLM_PUSH_REMOTE:-onecat}"
integration_branch="${VLLM_INTEGRATION_BRANCH:-main}"
owner="${VLLM_WORKTREE_OWNER:-agent}"
[[ "$owner" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || {
    echo "VLLM_WORKTREE_OWNER must be a valid branch component" >&2
    exit 2
}

if [[ -z "$base" ]]; then
    base="${push_remote}/${integration_branch}"
fi

timestamp="$(date -u +%Y%m%d-%H%M%S)"
branch="${owner}/sm70-${scope}-${timestamp}"
worktree_root="${VLLM_WORKTREE_ROOT:-$(dirname "$canonical_root")/worktrees}"
worktree="${worktree_root}/sm70-${scope}-${timestamp}"

if [[ "$dry_run" -eq 1 ]]; then
    printf 'CANONICAL=%s\nBASE=%s\nBRANCH=%s\nWORKTREE=%s\n' \
        "$canonical_root" "$base" "$branch" "$worktree"
    exit 0
fi

git -C "$canonical_root" remote get-url "$push_remote" >/dev/null
git -C "$canonical_root" status --short --branch
git -C "$canonical_root" fetch "$push_remote" --prune
git -C "$canonical_root" rev-parse --verify "${base}^{commit}" >/dev/null
git -C "$canonical_root" check-ref-format --branch "$branch" >/dev/null

mkdir -p "$worktree_root"
git -C "$canonical_root" worktree add -b "$branch" "$worktree" "$base"

base_sha="$(git -C "$worktree" rev-parse HEAD)"
printf 'BASE_SHA=%s\nBRANCH=%s\nWORKTREE=%s\n' \
    "$base_sha" "$branch" "$worktree"
git -C "$worktree" status --short --branch
