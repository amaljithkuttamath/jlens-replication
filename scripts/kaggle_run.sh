#!/usr/bin/env bash
# One-shot: push the Kaggle notebook, then poll until it finishes.
# Requires uv + a configured Kaggle token.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== Push =="
uv run jlens-kaggle push

echo
echo "== Poll (every 2 min) =="
while true; do
  out=$(uv run jlens-kaggle status 2>&1 || true)
  echo "$(date '+%H:%M:%S')  $out"
  case "$out" in
    *complete*|*Complete*|*COMPLETE*)
      echo "== Pull outputs =="
      uv run jlens-kaggle pull --out out
      break ;;
    *error*|*Error*|*ERROR*|*failed*|*Failed*|*cancel*)
      echo "Run failed"; exit 2 ;;
  esac
  sleep 120
done
