#!/usr/bin/env bash
#
# Publish the tail of a log as a GitHub check-run annotation.
#
#   scripts/ci-annotate.sh <logfile> [label]
#
# Why this exists: a failed step's log sits behind an authenticated endpoint (`GET
# /actions/jobs/<id>/logs` is 403 without a token), but the check run's **annotations** are public on a
# public repository. Vitest already produces readable annotations, which is how the tiny-fixture
# hardware drift was diagnosed (TASKS.md §10.14); a failing `build.sh`, `npm ci` or shell step produces
# only "Process completed with exit code 1", which says nothing. So those steps `tee` their output here
# and this turns the last lines into an annotation anyone — including a maintainer without credentials —
# can read from the Checks tab or the API.
#
# Workflow commands are line-oriented, so newlines inside a message are escaped as `%0A` (and `%` as
# `%25`, which must be escaped first). GitHub decodes those back into a multi-line annotation.
set -euo pipefail

LOG="${1:?usage: ci-annotate.sh <logfile> [label]}"
LABEL="${2:-step}"
LINES="${CI_ANNOTATE_LINES:-40}"

if [[ ! -f "$LOG" ]]; then
  echo "::warning::ci-annotate: no log at $LOG (the step may have failed before it wrote anything)"
  exit 0
fi

message="$(tail -n "$LINES" "$LOG")"
if [[ -z "$message" ]]; then
  echo "::warning::ci-annotate: $LOG is empty"
  exit 0
fi

message="${message//%/%25}"
message="${message//$'\r'/}"
message="${message//$'\n'/%0A}"

echo "::error title=${LABEL} failed (last ${LINES} lines)::$message"