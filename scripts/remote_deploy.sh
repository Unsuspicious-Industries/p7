#!/usr/bin/env bash
# Deploy the P7 demo (backend + frontend) to a remote server.
# - Rsync repo to remote
# - Builds frontend
# - Installs backend deps + Python package
# - Starts backend + frontend inside a tmux session on the remote host (no systemd) 

set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] user@host[:remote_path]

Options:
  -p, --port PORT           SSH port (default: 22)
  -i, --key PATH            SSH private key (optional)
  -d, --remote-dir DIR      Remote install parent dir (default: ~). Final install: DIR/p7
  --backend-port PORT       Backend port (default: 5001)
  --frontend-port PORT      Frontend port (default: 3000)
  --bootstrap               Attempt apt-based dependency install (requires sudo) 
  --watch, -w               Watch local files: rsync on change and reload remote services (requires 'fswatch')
  --dry-run                 Print commands only
  -h, --help                Show help

Notes:
  - The interactive watch mode requires 'fswatch' (install with 'brew install fswatch' on macOS).
  - Watch mode will rsync changed files, rebuild the frontend on the remote, and restart services.

Examples:
  ./scripts/remote_deploy.sh ubuntu@1.2.3.4
  ./scripts/remote_deploy.sh -w ubuntu@1.2.3.4
  ./scripts/remote_deploy.sh -d /scratch ubuntu@host
  ./scripts/remote_deploy.sh -p 2222 -i ~/.ssh/id_rsa ubuntu@host:/opt/p7
EOF
}

SSH_PORT=22
SSH_KEY=""
REMOTE_PARENT="~"
INSTALL_NAME="p7"
REMOTE_DIR=""
BACKEND_PORT=5001
FRONTEND_PORT=3000
BOOTSTRAP=0
DRY_RUN=0
WATCH=0

REMOTE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port) SSH_PORT="$2"; shift 2;;
    -i|--key) SSH_KEY="$2"; shift 2;;
    -d|--remote-dir) REMOTE_PARENT="$2"; shift 2;;
    --backend-port) BACKEND_PORT="$2"; shift 2;;
    --frontend-port) FRONTEND_PORT="$2"; shift 2;;
    --bootstrap) BOOTSTRAP=1; shift;;
    -w|--watch) WATCH=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    --) shift; break;;
    -*) echo "Unknown option: $1"; usage; exit 1;;
    *)
      if [[ -z "$REMOTE" ]]; then
        REMOTE="$1"; shift
      else
        echo "Multiple remote targets provided"; usage; exit 1
      fi
      ;;
  esac
done

if [[ -z "$REMOTE" ]]; then
  echo "Missing remote target (user@host[:path])" >&2
  usage
  exit 1
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

# If the caller didn't supply a full remote path, build it from the install parent
# provided with -d (default: ~) and append the install name (p7).
if [[ -z "${REMOTE_DIR:-}" ]]; then
  REMOTE_DIR="${REMOTE_PARENT%/}/$INSTALL_NAME"
fi

# Safety: only allow installs into non-system parent locations to keep things 'chill'
# Allowed parents: ~/, /home/*, /scratch, /tmp, /var/tmp, /opt
if [[ ! "$REMOTE_DIR" =~ ^(~\/|/home/|/scratch(/|$)|/tmp(/|$)|/var/tmp(/|$)|/opt(/|$)) ]]; then
  echo "Refusing to install to $REMOTE_DIR — choose a safe parent like ~/ or /scratch or /opt" >&2
  exit 1
fi

echo "Installing to: $REMOTE_DIR"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
)

SSH_OPTS=( -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=10 )
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPTS+=( -i "$SSH_KEY" )
fi

echo "Project root: $REPO_DIR"
echo "Target: $REMOTE_HOST:$REMOTE_DIR"

SSH_CMD_STR="ssh"
for opt in "${SSH_OPTS[@]}"; do
  SSH_CMD_STR+=" $opt"
done

RSYNC_CMD=( rsync -az --delete )
RSYNC_CMD+=( "${EXCLUDES[@]}" )
RSYNC_CMD+=( -e "$SSH_CMD_STR" )
RSYNC_CMD+=( "$REPO_DIR/" "$REMOTE_HOST:$REMOTE_DIR/" )

run_rsync() {
  echo "-> Running rsync to remote..."
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "${RSYNC_CMD[*]}"
    return
  fi
  "${RSYNC_CMD[@]}"
}

remote_reload() {
  echo "-> Triggering remote reload (rebuild frontend + restart services)..."
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY-RUN: ssh ${SSH_OPTS[*]} $REMOTE_HOST <reload-commands>"
    return
  fi

  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" bash -s -- "$REMOTE_DIR" "$BACKEND_PORT" "$FRONTEND_PORT" <<'END_RELOAD'
set -euo pipefail
REMOTE_DIR="$1"
BACKEND_PORT="$2"
FRONTEND_PORT="$3"

cd "$REMOTE_DIR"

# Kill any existing processes using the backend and frontend ports
echo "[remote] killing processes on ports $BACKEND_PORT and $FRONTEND_PORT..."
# Kill processes listening on the ports
if command -v fuser >/dev/null 2>&1; then
  fuser -k ${BACKEND_PORT}/tcp 2>/dev/null || true
  fuser -k ${FRONTEND_PORT}/tcp 2>/dev/null || true
elif command -v lsof >/dev/null 2>&1; then
  lsof -ti:${BACKEND_PORT} | xargs kill -9 2>/dev/null || true
  lsof -ti:${FRONTEND_PORT} | xargs kill -9 2>/dev/null || true
fi
# Kill any remaining gunicorn or serve processes
pkill -f gunicorn 2>/dev/null || true
pkill -f "serve -s build" 2>/dev/null || true
# Kill any existing tmux session
if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t "p7" 2>/dev/null || true
fi
sleep 2

# rebuild frontend (assumes node_modules already present from initial deploy)
if [ -d "visualization/frontend" ]; then
  echo "[remote] rebuilding frontend..."
  cd visualization/frontend
  npm run build || true
  cd "$REMOTE_DIR"
fi

P7_SESSION="p7"
LOG_DIR="$REMOTE_DIR/var/log"
mkdir -p "$LOG_DIR"

 BACKEND_CMD="$REMOTE_DIR/visualization/.venv/bin/gunicorn -w 1 --worker-class gthread --threads 4 --timeout 300 --graceful-timeout 30 -b 0.0.0.0:${BACKEND_PORT} app:app"
FRONTEND_CMD="cd $REMOTE_DIR/visualization/frontend && ./node_modules/.bin/serve -s build -l ${FRONTEND_PORT}"

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$P7_SESSION" 2>/dev/null; then
    tmux kill-window -t "$P7_SESSION":backend 2>/dev/null || true
    tmux new-window -t "$P7_SESSION" -n backend "cd $REMOTE_DIR/visualization/backend && $BACKEND_CMD 2>&1 | tee $LOG_DIR/backend.log"
    tmux kill-window -t "$P7_SESSION":frontend 2>/dev/null || true
    tmux new-window -t "$P7_SESSION" -n frontend "$FRONTEND_CMD 2>&1 | tee $LOG_DIR/frontend.log"
  else
    tmux new-session -d -s "$P7_SESSION" -n backend "cd $REMOTE_DIR/visualization/backend && $BACKEND_CMD 2>&1 | tee $LOG_DIR/backend.log"
    tmux new-window -t "$P7_SESSION" -n frontend "$FRONTEND_CMD 2>&1 | tee $LOG_DIR/frontend.log"
  fi
else
  pkill -f gunicorn 2>/dev/null || true
  nohup bash -c "cd $REMOTE_DIR/visualization/backend && $BACKEND_CMD" > "$LOG_DIR/backend.log" 2>&1 &
  pkill -f "node_modules/.bin/serve -s build" 2>/dev/null || true
  nohup bash -c "$FRONTEND_CMD" > "$LOG_DIR/frontend.log" 2>&1 &
fi

sleep 1
echo "[remote] reloaded backend and frontend"
END_RELOAD
}

start_watch() {
  trap 'echo "[watch] interrupted — exiting"; exit 0' INT TERM
  if command -v fswatch >/dev/null 2>&1; then
    echo "[watch] watching $REPO_DIR (excludes: target, .git, node_modules, __pycache__, .venv)"
    fswatch -o -r --exclude '(^|/)(target|\.git|node_modules|__pycache__|\.venv)(/|$)' "$REPO_DIR" | while read -r _; do
      echo "[watch] change detected — rsync + reload"
      run_rsync
      remote_reload
    done
  else
    echo "fswatch not found. Install with: brew install fswatch" >&2
    exit 1
  fi
}

# perform initial rsync (and later the watch will rsync on changes)
run_rsync

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY-RUN: ssh ${SSH_OPTS[*]} $REMOTE_HOST mkdir -p '$REMOTE_DIR'"
  echo "DRY-RUN: ssh ${SSH_OPTS[*]} $REMOTE_HOST bash -s -- ..."
  exit 0
fi

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" mkdir -p "$REMOTE_DIR"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" bash -s -- \
  "$REMOTE_DIR" "$BACKEND_PORT" "$FRONTEND_PORT" "$BOOTSTRAP" <<'END_REMOTE'
set -euo pipefail
REMOTE_DIR="$1"
BACKEND_PORT="$2"
FRONTEND_PORT="$3"
BOOTSTRAP="$4"

cd "$REMOTE_DIR"
echo "[remote] pwd=$(pwd) host=$(hostname)"

if [[ "$BOOTSTRAP" == "1" ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip nodejs npm
  fi
fi

PY_BIN=""
for c in python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then PY_BIN="$c"; break; fi
done
if [[ -z "$PY_BIN" ]]; then
  echo "[remote] python not found (use --bootstrap or install python3)" >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "[remote] npm not found (use --bootstrap or install nodejs/npm)" >&2
  exit 1
fi

VENV_DIR="visualization/.venv"
"$PY_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel maturin

# Ensure we're in the right directory
echo "[remote] working directory: $(pwd)"
echo "[remote] listing files..."
ls -la | head -10

# Build and install the Rust extension with maturin
echo "[remote] building Rust extension with maturin..."
if [ -f "pyproject.toml" ]; then
  if maturin develop --release; then
    echo "[remote] maturin build successful"
  else
    echo "[remote] maturin build failed, trying pip install..."
    python -m pip install -e .
  fi
else
  echo "[remote] pyproject.toml not found, using pip install..."
  python -m pip install -e .
fi

echo "[remote] installing backend requirements..."
echo "[remote] current directory: $(pwd)"
echo "[remote] checking if requirements.txt exists..."
ls -la visualization/backend/requirements.txt 2>/dev/null || {
  echo "[remote] ERROR: requirements.txt not found at expected location"
  echo "[remote] looking for it..."
  find . -name "requirements.txt" -path "*/backend/*" 2>/dev/null | head -5
  exit 1
}
python -m pip install -r visualization/backend/requirements.txt
python -m pip install gunicorn

# Kill any existing processes using the backend and frontend ports
echo "[remote] killing existing processes on ports $BACKEND_PORT and $FRONTEND_PORT..."
if command -v fuser >/dev/null 2>&1; then
  fuser -k ${BACKEND_PORT}/tcp 2>/dev/null || true
  fuser -k ${FRONTEND_PORT}/tcp 2>/dev/null || true
elif command -v lsof >/dev/null 2>&1; then
  lsof -ti:${BACKEND_PORT} | xargs kill -9 2>/dev/null || true
  lsof -ti:${FRONTEND_PORT} | xargs kill -9 2>/dev/null || true
fi
pkill -f gunicorn 2>/dev/null || true
pkill -f "serve -s build" 2>/dev/null || true
tmux kill-session -t "p7" 2>/dev/null || true
sleep 2

# Verify the package is properly installed
echo "[remote] verifying proposition_7 installation..."
python -c "import proposition_7; print(f'proposition_7 version: {proposition_7.__version__}')" || {
  echo "[remote] ERROR: proposition_7 module not found or broken"
  echo "[remote] Attempting to rebuild..."
  maturin develop --release
  python -c "import proposition_7; print(f'proposition_7 version: {proposition_7.__version__}')" || {
    echo "[remote] FATAL: Could not install proposition_7"
    exit 1
  }
}

# Install optional ML dependencies for constrained generation
echo "[remote] installing ML dependencies (transformers, torch, accelerate, bitsandbytes)..."
python -m pip install transformers torch accelerate bitsandbytes || echo "[remote] warning: failed to install ML dependencies"

cd visualization/frontend
# Prefer a reproducible install when a lockfile exists; otherwise fall back to legacy install.
if [ -f package-lock.json ]; then
  npm ci --legacy-peer-deps || npm install --legacy-peer-deps
else
  npm install --legacy-peer-deps
  # create a package-lock on the remote for reproducible builds (won't be committed)
  npm install --package-lock-only --legacy-peer-deps || true
fi
npm install --no-save serve
npm run build
# Deploy using tmux (no systemd)
P7_SESSION="p7"
LOG_DIR="$REMOTE_DIR/var/log"
mkdir -p "$LOG_DIR"

# replace existing session if present
tmux kill-session -t "$P7_SESSION" 2>/dev/null || true

# start backend in tmux (or background if tmux missing)
# Use longer timeouts and fewer workers to avoid model load timeouts/OOM
BACKEND_CMD="$REMOTE_DIR/visualization/.venv/bin/gunicorn -w 1 -b 0.0.0.0:${BACKEND_PORT} --timeout 600 --graceful-timeout 600 --keep-alive 5 app:app"
if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "$P7_SESSION" -n backend "cd $REMOTE_DIR/visualization/backend && $BACKEND_CMD 2>&1 | tee $LOG_DIR/backend.log"
else
  nohup bash -c "cd $REMOTE_DIR/visualization/backend && $BACKEND_CMD" > "$LOG_DIR/backend.log" 2>&1 &
fi

# start frontend in tmux (or background if tmux missing)
FRONTEND_CMD="cd $REMOTE_DIR/visualization/frontend && ./node_modules/.bin/serve -s build -l ${FRONTEND_PORT}"
if command -v tmux >/dev/null 2>&1; then
  tmux new-window -t "$P7_SESSION" -n frontend "$FRONTEND_CMD 2>&1 | tee $LOG_DIR/frontend.log"
else
  nohup bash -c "$FRONTEND_CMD" > "$LOG_DIR/frontend.log" 2>&1 &
fi

sleep 1

echo "[remote] demo deployed (tmux session: $P7_SESSION)"
echo "[remote] attach with: tmux attach -t $P7_SESSION" 
echo "[remote] backend logs: $LOG_DIR/backend.log"
echo "[remote] frontend logs: $LOG_DIR/frontend.log"
END_REMOTE

echo "Deploy complete."

if [[ "$WATCH" == "1" ]]; then
  echo "[local] starting interactive watch — will rsync on changes and reload remote services (Ctrl-C to stop)"
  start_watch
fi
