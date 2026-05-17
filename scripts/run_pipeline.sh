#!/usr/bin/env bash
# Example end-to-end pipeline for AID few-shot 16-shot.
set -euo pipefail

DATA_ROOT="data/AID"
DESC_JSON="data/llm_cache/aid.json"
CACHE_PATH="data/cache/aid_teacher.pt"

# 1) generate per-class descriptions with Gemini (or hand-author the JSON)
python scripts/generate_descriptions.py --data-root "$DATA_ROOT" --out "$DESC_JSON" --n 30

# 2) cache teacher visual + selective text prototypes
python -m avion.offline \
  --data-root "$DATA_ROOT" \
  --descriptions "$DESC_JSON" \
  --out "$CACHE_PATH" \
  --teacher-backbone ViT-H-14 \
  --teacher-pretrained laion2b_s32b_b79k

# 3) train the student
python -m avion.train --config configs/few_shot_aid.yaml

# 4) evaluate
python -m avion.evaluate --config configs/few_shot_aid.yaml --task few_shot
