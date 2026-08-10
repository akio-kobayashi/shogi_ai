#!/bin/sh
rsync -avz --prune-empty-dirs \
  --include='*/' \
  --include='move_metrics.json' \
  --include='token_probe_metrics.json' \
  --include='run_manifest.json' \
  --include='training_history.json' \
  --include='probe_metrics.json' \
  --include='probe_metrics_detail.json' \
  --include='linear_probes.pt' \
  --include='artifact_verification.json' \
  --exclude='*' \
  akio@100.74.3.9:/home/akio/GitHub/shogi_ai/shogi_state_tracking/llama_results/ \
  collected_results/llama_machine/
