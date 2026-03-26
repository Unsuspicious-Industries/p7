#!/usr/bin/env bash
# Run the benchmark suite on a remote GPU server.
#
# What this script does:
# 1) rsyncs the current repo to the remote machine
# 2) creates/uses a remote virtualenv
# 3) builds and installs p7 (maturin develop --release)
# 4) runs mk.py, run.py, agg.py with configurable flags
# 5) optionally runs detached in tmux and optionally pulls results locally

set -euo pipefail
IFS=$'\n\t'

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] user@host[:remote_path]

Options:
  -p, --port PORT            SSH port (default: 22)
  -i, --key PATH             SSH private key (optional)
  -c, --ssh-cmd "SSH ..."    Extra ssh command/params (supports -L/-J/etc)
  --batch                    Force non-interactive SSH (BatchMode=yes)
  -d, --remote-dir DIR       Remote install parent dir (default: ~). Final path: DIR/p7
  --run-id ID                Run id for output folder (default: timestamp)
  --tasks LIST               Bench task sets (default: stlc,fun,imp,spec)
  --models LIST              Comma-separated HF model ids (default: run.py default set)
  --tries N                  Number of tries per task/mode (default: 3)
  --max-tasks N              Max tasks per suite (default: 0 = all)
  --device DEVICE            Torch device (default: cuda)
  --feed-only                Use feed-only constrained decoding
  --skip-mk                  Skip regenerating benchmark CSVs
  --bootstrap                Try apt-based bootstrap + rustup install if missing
  --detached                 Run remotely in tmux and return immediately
  --tmux-session NAME        tmux session name for detached mode (default: p7-bench)
  --no-sync                  Do not rsync local repo before running
  --no-pull                  Do not pull results back to local benchmarks/out
  --dry-run                  Print commands without executing
  -h, --help                 Show this help

Examples:
  ./scripts/remote_benchmark.sh ubuntu@gpu-box
  ./scripts/remote_benchmark.sh --device cuda --tries 1 --max-tasks 10 ubuntu@gpu-box
  ./scripts/remote_benchmark.sh -c "ssh -p 19241 root@175.155.64.231 -L 8080:localhost:8080" -d /scratch
  ./scripts/remote_benchmark.sh --detached --tmux-session bench-a100 ubuntu@gpu-box:/scratch/p7
EOF
}

SSH_PORT=22
USER_SET_PORT=0
SSH_KEY=""
SSH_CUSTOM=""
SSH_CUSTOM_HOST=""
REMOTE_PARENT="~"
INSTALL_NAME="p7"
REMOTE_DIR=""
RUN_ID="$(date +%Y%m%d_%H%M%S)"
TASKS="stlc,fun,imp,spec"
MODELS=""
TRIES=3
MAX_TASKS=0
DEVICE="cuda"
FEED_ONLY=0
SKIP_MK=0
BOOTSTRAP=0
DETACHED=0
TMUX_SESSION="p7-bench"
SYNC=1
PULL=1
DRY_RUN=0
BATCH_MODE=0

REMOTE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port) SSH_PORT="$2"; USER_SET_PORT=1; shift 2 ;;
    -i|--key) SSH_KEY="$2"; shift 2 ;;
    -c|--ssh-cmd) SSH_CUSTOM="$2"; shift 2 ;;
    -d|--remote-dir) REMOTE_PARENT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --tries) TRIES="$2"; shift 2 ;;
    --max-tasks) MAX_TASKS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --feed-only) FEED_ONLY=1; shift ;;
    --skip-mk) SKIP_MK=1; shift ;;
    --bootstrap) BOOTSTRAP=1; shift ;;
    --detached) DETACHED=1; shift ;;
    --tmux-session) TMUX_SESSION="$2"; shift 2 ;;
    --no-sync) SYNC=0; shift ;;
    --no-pull) PULL=0; shift ;;
    --batch) BATCH_MODE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    *)
      if [[ -z "$REMOTE" ]]; then
        REMOTE="$1"
        shift
      else
        echo "Multiple remote targets provided" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

SSH_CUSTOM_ARGS=()
SSH_CUSTOM_HAS_PORT=0
if [[ -n "$SSH_CUSTOM" ]]; then
  # Accept either:
  #   -c "-p 19241 -L 8080:localhost:8080"
  # or
  #   -c "ssh -p 19241 root@host -L 8080:localhost:8080"
  old_ifs="$IFS"
  IFS=' '
  read -r -a _raw <<< "$SSH_CUSTOM"
  IFS="$old_ifs"
  idx=0
  if [[ ${#_raw[@]} -gt 0 ]]; then
    first="${_raw[0]}"
    if [[ "$first" == ":ssh" || "$first" == "ssh" ]]; then
      idx=1
    fi
  fi

  expect_value=0
  while [[ $idx -lt ${#_raw[@]} ]]; do
    tok="${_raw[$idx]}"
    if [[ "$expect_value" == "1" ]]; then
      SSH_CUSTOM_ARGS+=("$tok")
      expect_value=0
      idx=$((idx + 1))
      continue
    fi

    case "$tok" in
      -b|-c|-D|-E|-e|-F|-I|-i|-J|-L|-l|-m|-O|-o|-p|-Q|-R|-S|-W|-w)
        SSH_CUSTOM_ARGS+=("$tok")
        if [[ "$tok" == "-p" ]]; then
          SSH_CUSTOM_HAS_PORT=1
        fi
        expect_value=1
        idx=$((idx + 1))
        ;;
      -p*)
        SSH_CUSTOM_ARGS+=("$tok")
        SSH_CUSTOM_HAS_PORT=1
        idx=$((idx + 1))
        ;;
      --)
        idx=$((idx + 1))
        ;;
      -*)
        SSH_CUSTOM_ARGS+=("$tok")
        idx=$((idx + 1))
        ;;
      *)
        # Treat the first standalone token as a host; keep using positional
        # remote path if provided separately.
        if [[ -z "$SSH_CUSTOM_HOST" ]]; then
          SSH_CUSTOM_HOST="$tok"
        else
          SSH_CUSTOM_ARGS+=("$tok")
        fi
        idx=$((idx + 1))
        ;;
    esac
  done
fi

if [[ -z "$REMOTE" ]]; then
  if [[ -n "$SSH_CUSTOM_HOST" ]]; then
    REMOTE="$SSH_CUSTOM_HOST"
  else
    echo "Missing remote target (user@host[:path])" >&2
    usage
    exit 1
  fi
fi

if [[ -n "$REMOTE" && -n "$SSH_CUSTOM_HOST" && ( "$REMOTE" == /* || "$REMOTE" == ~/* ) ]]; then
  REMOTE_DIR="$REMOTE"
  REMOTE="$SSH_CUSTOM_HOST"
fi

if [[ "$REMOTE" == *:* ]]; then
  REMOTE_HOST="${REMOTE%%:*}"
  REMOTE_PATH="${REMOTE#*:}"
  if [[ -n "$REMOTE_PATH" ]]; then
    REMOTE_DIR="$REMOTE_PATH"
  fi
else
  REMOTE_HOST="$REMOTE"
fi

if [[ -z "${REMOTE_DIR:-}" ]]; then
  REMOTE_DIR="${REMOTE_PARENT%/}/$INSTALL_NAME"
fi

if [[ ! "$REMOTE_DIR" =~ ^(~/|/home/|/scratch(/|$)|/tmp(/|$)|/var/tmp(/|$)|/opt(/|$)) ]]; then
  echo "Refusing to use unsafe remote dir: $REMOTE_DIR" >&2
  echo "Choose a path under ~, /home, /scratch, /tmp, /var/tmp, or /opt." >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_OUT_DIR="$REPO_DIR/benchmarks/out/$RUN_ID"
REMOTE_OUT_DIR="$REMOTE_DIR/benchmarks/out/$RUN_ID"

SSH_OPTS=( -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 )
if [[ "$SSH_CUSTOM_HAS_PORT" == "0" || "$USER_SET_PORT" == "1" ]]; then
  SSH_OPTS+=( -p "$SSH_PORT" )
fi
if [[ "$BATCH_MODE" == "1" ]]; then
  SSH_OPTS+=( -o BatchMode=yes )
fi
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPTS+=( -i "$SSH_KEY" )
fi
if [[ ${#SSH_CUSTOM_ARGS[@]} -gt 0 ]]; then
  SSH_OPTS+=( "${SSH_CUSTOM_ARGS[@]}" )
fi

SSH_CMD_STR="ssh"
for opt in "${SSH_OPTS[@]}"; do
  SSH_CMD_STR+=" $opt"
done

EXCLUDES=(
  --exclude 'target'
  --exclude '*/target'
  --exclude '.git'
  --exclude '.git/*'
  --exclude 'node_modules'
  --exclude '__pycache__'
  --exclude '*.pyc'
  --exclude '*.pyo'
  --exclude '.venv'
  --exclude 'benchmarks/out'
)

RSYNC_CMD=( rsync -az --delete --info=progress2,stats2 )
RSYNC_CMD+=( "${EXCLUDES[@]}" )
RSYNC_CMD+=( -e "$SSH_CMD_STR" )
RSYNC_CMD+=( "$REPO_DIR/" "$REMOTE_HOST:$REMOTE_DIR/" )

log "Remote host:      $REMOTE_HOST"
log "Remote directory: $REMOTE_DIR"
log "Run id:           $RUN_ID"
log "Device:           $DEVICE"
log "Tasks:            $TASKS"
log "Tries:            $TRIES"
log "Max tasks:        $MAX_TASKS"
if [[ -n "$MODELS" ]]; then
  log "Models:           $MODELS"
else
  log "Models:           <run.py defaults>"
fi
log "Detached:         $DETACHED"

if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY-RUN enabled"
fi

if [[ "$SYNC" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY-RUN: ssh ${SSH_OPTS[*]} $REMOTE_HOST mkdir -p '$REMOTE_DIR'"
    echo "DRY-RUN: ${RSYNC_CMD[*]}"
  else
    log "[1/5] Checking SSH connectivity"
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "echo '[remote] SSH connected: '"'"'$(hostname)'"'"'"
    log "[2/5] Ensuring remote directory exists"
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" mkdir -p "$REMOTE_DIR"
    log "[3/5] Syncing repository to remote (rsync progress below)"
    "${RSYNC_CMD[@]}"
    log "[3/5] Sync complete"
  fi
fi

RUN_CMD="python benchmarks/run.py --tasks '$TASKS' --tries '$TRIES' --max-tasks '$MAX_TASKS' --device '$DEVICE' --out '$REMOTE_OUT_DIR/raw.jsonl' --trace-out '$REMOTE_OUT_DIR/trace.jsonl'"
if [[ -n "$MODELS" ]]; then
  RUN_CMD+=" --models '$MODELS'"
fi
if [[ "$FEED_ONLY" == "1" ]]; then
  RUN_CMD+=" --feed-only"
fi

MK_CMD="python benchmarks/mk.py"
if [[ "$SKIP_MK" == "1" ]]; then
  MK_CMD=":"
fi

REMOTE_WORKLOAD=$(cat <<EOF
set -euo pipefail
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[remote][%s] %s\\n' "\$(ts)" "\$*"; }
cd "$REMOTE_DIR"
mkdir -p "$REMOTE_OUT_DIR"
export PYTHONUNBUFFERED=1

log "Starting remote benchmark workload"
log "Working directory: \$(pwd)"
log "Output directory: $REMOTE_OUT_DIR"

PY_BIN=""
for c in python3.11 python3 python; do
  if command -v "\$c" >/dev/null 2>&1; then PY_BIN="\$c"; break; fi
done
if [[ -z "\$PY_BIN" ]]; then
  echo "[remote] python not found" >&2
  exit 1
fi
log "Using Python: \$PY_BIN"

if [[ "$BOOTSTRAP" == "1" ]]; then
  log "Bootstrap enabled; checking apt-get"
  if command -v apt-get >/dev/null 2>&1; then
    log "Installing base build dependencies via apt"
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip build-essential pkg-config curl
  fi
fi

if ! command -v rustc >/dev/null 2>&1; then
  if [[ "$BOOTSTRAP" == "1" ]]; then
    log "Rust not found; installing via rustup"
    curl https://sh.rustup.rs -sSf | sh -s -- -y
    source "\$HOME/.cargo/env"
  else
    echo "[remote] rustc not found; rerun with --bootstrap or install Rust" >&2
    exit 1
  fi
fi

if [[ ! -d ".venv-bench" ]]; then
  log "Creating .venv-bench"
  "\$PY_BIN" -m venv .venv-bench
fi
source .venv-bench/bin/activate
log "Activated virtualenv"

log "Installing Python build dependencies"
python -m pip install --upgrade pip setuptools wheel maturin
log "Building/installing p7 extension (this can take a while)"
maturin develop --release --target-dir target || python -m pip install -e .
log "Installing ML dependencies (torch/transformers/etc)"
python -m pip install --upgrade transformers torch accelerate bitsandbytes

log "Generating benchmark task CSVs"
$MK_CMD
log "Running benchmark suite (model load can take several minutes before first task output)"
$RUN_CMD
log "Aggregating benchmark outputs"
python benchmarks/agg.py --in "$REMOTE_OUT_DIR/raw.jsonl" --out-dir "$REMOTE_OUT_DIR"

log "Finished benchmark run"
log "Outputs: $REMOTE_OUT_DIR"
EOF
)

if [[ "$DETACHED" == "1" ]]; then
  WORKLOAD_B64="$(printf '%s' "$REMOTE_WORKLOAD" | base64 | tr -d '\n')"
  DETACHED_CMD="mkdir -p '$REMOTE_OUT_DIR' && printf '%s' '$WORKLOAD_B64' | base64 -d > '$REMOTE_OUT_DIR/run.sh' && chmod +x '$REMOTE_OUT_DIR/run.sh' && tmux new-session -d -s '$TMUX_SESSION' \"bash '$REMOTE_OUT_DIR/run.sh' 2>&1 | tee '$REMOTE_OUT_DIR/remote.log'\""
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY-RUN: ssh ${SSH_OPTS[*]} $REMOTE_HOST $DETACHED_CMD"
  else
    log "[4/5] Starting detached remote run in tmux session '$TMUX_SESSION'"
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$DETACHED_CMD"
    log "[5/5] Detached run launched"
    log "Tail logs:   ssh ${SSH_OPTS[*]} $REMOTE_HOST 'tail -f $REMOTE_OUT_DIR/remote.log'"
    log "Attach tmux: ssh ${SSH_OPTS[*]} $REMOTE_HOST 'tmux attach -t $TMUX_SESSION'"
    log "When done, pull results with:"
    log "  rsync -az -e \"$SSH_CMD_STR\" $REMOTE_HOST:$REMOTE_OUT_DIR/ $LOCAL_OUT_DIR/"
  fi
  exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY-RUN: ssh ${SSH_OPTS[*]} $REMOTE_HOST 'bash -s' <<'EOF'"
  echo "$REMOTE_WORKLOAD"
  echo "EOF"
else
  log "[4/5] Running remote benchmark workload (streaming remote progress)"
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" bash -s <<EOF
$REMOTE_WORKLOAD
EOF
  log "[4/5] Remote workload completed"
fi

if [[ "$PULL" == "1" ]]; then
  mkdir -p "$LOCAL_OUT_DIR"
  PULL_CMD=( rsync -az -e "$SSH_CMD_STR" "$REMOTE_HOST:$REMOTE_OUT_DIR/" "$LOCAL_OUT_DIR/" )
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY-RUN: ${PULL_CMD[*]}"
  else
    log "[5/5] Pulling results back to local machine"
    "${PULL_CMD[@]}"
    log "Pulled results to: $LOCAL_OUT_DIR"
  fi
fi

log "Done."
