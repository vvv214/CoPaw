#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <host> <remote-path>" >&2
  exit 1
fi

host="$1"
remote_path="$2"
timestamp="$(date +%Y%m%dT%H%M%S)"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

ssh -o BatchMode=yes -o ConnectTimeout=10 "${host}" \
  "mkdir -p '${remote_path}' && \
   if [ -f '${remote_path}/.git' ] && \
      grep -q '/Users/.*/.git/worktrees/' '${remote_path}/.git'; then \
     mv '${remote_path}/.git' \
        '${remote_path}/.git.broken-mac-worktree-${timestamp}'; \
   fi"

rsync -az --delete \
  --exclude='.git' \
  --exclude='.venv/' \
  --exclude='.venv*/' \
  --exclude='node_modules/' \
  --exclude='console/node_modules/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='.claude/' \
  --exclude='.chat-demo/' \
  --exclude='.chat-demo.secret/' \
  --exclude='*.launch.log' \
  --exclude='docs/run_watch/' \
  "${repo_root}/" \
  "${host}:${remote_path}/"

