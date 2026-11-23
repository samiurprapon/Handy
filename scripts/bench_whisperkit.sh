#!/bin/zsh
# WhisperKit vs transcribe-cpp benchmark on this machine (Apple M2).
# Usage: scripts/bench_whisperkit.sh <handy-binary> [whisper-gguf-model-id]
# Clips must exist as /tmp/bench_{5,15,30,60}s.wav (16 kHz mono PCM16).

set -euo pipefail
BIN="${1:?handy binary}"
GGUF_MODEL="${2:-}"

for secs in 5 15 30 60; do
  wav="/tmp/bench_${secs}s.wav"
  [[ -f "$wav" ]] || { echo "missing $wav"; exit 1 }
  echo "== WhisperKit, ${secs}s =="
  "$BIN" --transcribe-file "$wav" --model whisperkit/bengaliai-whisper-medium --json --repeat 3
  if [[ -n "$GGUF_MODEL" ]]; then
    echo "== transcribe-cpp ($GGUF_MODEL), ${secs}s =="
    "$BIN" --transcribe-file "$wav" --model "$GGUF_MODEL" --json --repeat 3
  fi
done
