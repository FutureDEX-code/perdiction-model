#!/bin/zsh
set -euo pipefail
DIR="/Users/jasperZ@bixin.com/Documents/Prediction-model-ensemble/polymarket/polymarket-temperature"
cd "$DIR"
if [ ! -d .venv ]; then
  /opt/homebrew/bin/python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -U pip
pip install -q -r requirements.txt
# stop old
if [ -f server.pid ]; then
  kill "$(cat server.pid)" 2>/dev/null || true
  rm -f server.pid
fi
lsof -tiTCP:16888 -sTCP:LISTEN | xargs kill 2>/dev/null || true
nohup python server.py > server.log 2>&1 &
echo $! > server.pid
sleep 1
curl -sS "http://127.0.0.1:16888/health" || true
echo
echo "PID $(cat server.pid)"
