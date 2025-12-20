#!/usr/bin/env bash
set -euo pipefail

# 用法
if [ "$#" -lt 2 ]; then
  echo "Usage:"
  echo "  $0 <tmux_session_name> <configs_subdir> [adalab args...]"
  echo
  echo "Example:"
  echo "  $0 adalab_overfit overfit_configs \\"
  echo "     --experiments-dir batch_exp/overfit \\"
  echo "     --viz"
  exit 1
fi

SESSION_NAME="$1"
CONFIG_SUBDIR="$2"
shift 2
ADALAB_ARGS=("$@")

BASE_DIR="$(pwd)"
CONFIG_DIR="$BASE_DIR/configs/$CONFIG_SUBDIR"

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Error: directory not found: $CONFIG_DIR"
  exit 1
fi

# 初始化 conda
CONDA_BASE="$HOME/miniconda3"

if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "Error: conda.sh not found under $CONDA_BASE"
  exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

# tmux（使用 conda base 中的 tmux）
TMUX_BIN="$CONDA_BASE/bin/tmux"

if [ ! -x "$TMUX_BIN" ]; then
  echo "Error: tmux not found at $TMUX_BIN"
  exit 1
fi

# 收集配置文件
mapfile -t CONFIGS < <(
  find "$CONFIG_DIR" -maxdepth 1 -type f -name "*.json" | sort
)

if [ "${#CONFIGS[@]}" -eq 0 ]; then
  echo "Error: no json configs found in $CONFIG_DIR"
  exit 1
fi

# 创建 tmux session
if "$TMUX_BIN" has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Error: tmux session '$SESSION_NAME' already exists"
  exit 1
fi

"$TMUX_BIN" new-session -d -s "$SESSION_NAME"

# 每个 config 一个 window
for cfg in "${CONFIGS[@]}"; do
  cfg_name="$(basename "$cfg" .json)"

  "$TMUX_BIN" new-window -t "$SESSION_NAME" -n "$cfg_name"

  "$TMUX_BIN" send-keys -t "$SESSION_NAME:$cfg_name" \
    "cd \"$BASE_DIR\" && \
     source \"$CONDA_BASE/etc/profile.d/conda.sh\" && \
     conda activate adalab && \
     echo \"[Env] \$CONDA_PREFIX\" && \
     echo \"[Config] $cfg\" && \
     adalab --config \"$cfg\" ${ADALAB_ARGS[*]}" C-m
done

# 删除 tmux 自动生成的 window 0
"$TMUX_BIN" kill-window -t "$SESSION_NAME:0"

# 进入 session
"$TMUX_BIN" attach -t "$SESSION_NAME"
