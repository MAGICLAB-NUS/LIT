#!/usr/bin/env bash
# Paths every entry point needs. Export these in your shell, or copy this file to
# env.local.sh (git-ignored), edit it, and the entry points will pick it up.
#
#   LIT_MOLMOACT2     the Molmoact2 checkout, cloned with --recursive
#   LIBERO_PLUS_ROOT  the LIBERO-plus checkout (contains libero/libero/benchmark/)
#   DATASET_ROOT      LIBERO in LeRobot format — training only
#   LIT_WORK          where logs and aggregated json are written

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${_here}/env.local.sh" ] && . "${_here}/env.local.sh"

: "${LIT_MOLMOACT2:=}"
: "${LIBERO_PLUS_ROOT:=}"
: "${DATASET_ROOT:=}"
: "${LIT_WORK:=${_here}/../_work}"

lit_need() {   # lit_need VAR "what it is" [file that must exist inside]
  local var="$1" desc="$2" probe="${3:-}"
  local val="${!var:-}"
  if [ -z "$val" ] || [ ! -d "$val" ]; then
    echo "error: \$$var is not set to an existing directory ($desc)." >&2
    echo "       export $var=/path/to/... , or set it in scripts/env.local.sh" >&2
    return 1
  fi
  if [ -n "$probe" ] && [ ! -e "$val/$probe" ]; then
    echo "error: \$$var=$val does not contain $probe — is that the right directory?" >&2
    return 1
  fi
  return 0
}

lit_need_eval() {
  lit_need LIT_MOLMOACT2 "the Molmoact2 checkout" "scripts/libero_eval" || return 1
  lit_need LIBERO_PLUS_ROOT "the LIBERO-plus checkout" "libero/libero/benchmark/task_classification.json" || return 1
  mkdir -p "$LIT_WORK"
}

lit_need_train() {
  lit_need_eval || return 1
  lit_need DATASET_ROOT "LIBERO in LeRobot format" || return 1
}
