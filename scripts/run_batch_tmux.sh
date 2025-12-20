#!/usr/bin/env bash
set -euo pipefail

# 参数解析
if [ "$#" -lt 2 ]; then
  echo "Usage:"
  echo "  $0 <tmux_session_name> <configs_subdir> [adalab args...]"
  echo
  echo "Example:"
  echo "  $0 adalab_baseline baseline_configs \\"
  echo "     --experiments-dir experiments/baseline \\"
  echo "     --course-folder data/test_images \\"
  echo "     --viz"
  exit 1
fi

SESSION_NAME="$1"
CONFIG_SUBDIR="$2"
shift 2
ADALAB_ARGS=("$@") # 用数组，避免空格/引号问题

BASE_DIR="$(pwd)"
CONFIG_DIR="$BASE_DIR/configs/$CONFIG_SUBDIR"

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Error: directory not found: $CONFIG_DIR"
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
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Error: tmux session '$SESSION_NAME' already exists"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME"

# 每个 config 一个 window
for cfg in "${CONFIGS[@]}"; do
  cfg_name="$(basename "$cfg" .json)"

  tmux new-window -t "$SESSION_NAME" -n "$cfg_name"

  tmux send-keys -t "$SESSION_NAME:$cfg_name" \
    "cd $BASE_DIR && \
         echo '[Config]' $cfg && \
         adalab --config $cfg ${ADALAB_ARGS[*]}" C-m
done

# 删除 tmux 自动创建的空 window
tmux kill-window -t "$SESSION_NAME:0"

# 进入 session
tmux attach -t "$SESSION_NAME"
