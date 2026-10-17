#!/usr/bin/env bash
# Train every feature-set variant end to end on the GPU machine and write one Kaggle
# submission per variant, plus a "bag" blend of all models.
#
#   fs1  129 selected + forecast                                  (current best set)
#   fs2  re-selected from base + forecast + dev + app2 + freq + peer + shape
#   fs3  all 255 base features + forecast, no selection
#   bag  every model from fs1-fs3 blended together
#
# Usage:  bash run_variants.sh [stage ...]        stages: prep select tune train blend
#         bash run_variants.sh                    (all stages, in order)
#         N_TRIALS=60 CAT_TRIALS=30 bash run_variants.sh tune
#         VARIANTS="fs2 fs3" bash run_variants.sh train   (only these variants; blend always
#                                                         blends whichever preds exist)
# Stages can be rerun independently; tuning resumes from cache/optuna.db.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

N_TRIALS=${N_TRIALS:-40}      # Optuna trials per XGBoost model per feature set
CAT_TRIALS=${CAT_TRIALS:-25}  # Optuna trials for CatBoost per feature set
POOL="forecast dev app2 freq peer shape"

declare -A FEATS=([fs1]="selected" [fs2]="sel_v2" [fs3]="all")
declare -A EXTRA=([fs1]="forecast" [fs2]="$POOL" [fs3]="forecast")
VARIANTS=${VARIANTS:-fs1 fs2 fs3}   # e.g. VARIANTS="fs2 fs3" to skip fs1

log() { echo "[$(date '+%H:%M:%S')] $*"; }
run() { local name=$1; shift; log "$name: $*"; "$@" 2>&1 | tee "logs/$name.log"; }

stage_prep() {
    [ -f cache/features.pkl ] || run features python features.py
    [ -f cache/forecast_features.pkl ] || run forecast python forecast.py
    run families python fe_lab.py --build-only
}

stage_select() {
    run select_sel_v2 python select_features.py 30 --extra $POOL --out sel_v2
}

stage_tune() {
    for v in $VARIANTS; do
        for m in xgbl xgb; do
            run "tune_${m}_${v}" python tune.py $m "$N_TRIALS" "${FEATS[$v]}" ${EXTRA[$v]}
        done
        run "tune_cat_${v}" python tune.py cat "$CAT_TRIALS" "${FEATS[$v]}" ${EXTRA[$v]}
    done
}

stage_train() {
    for v in $VARIANTS; do
        f=${FEATS[$v]}; e=${EXTRA[$v]}
        run "train_xgbl_${v}" python train_gbm.py xgbl --repeats 5 --feats "$f" --extra $e --tag "xgbl_${v}"
        run "train_xgb_${v}"  python train_gbm.py xgb  --repeats 3 --feats "$f" --extra $e --tag "xgb_${v}"
        run "train_cat_${v}"  python train_gbm.py cat  --repeats 3 --feats "$f" --extra $e --tag "cat_${v}"
        run "train_tabm_${v}" python train_tabm.py     --repeats 3 --feats "$f" --extra $e --tag "tabm_${v}"
    done
}

stage_blend() {
    all=""
    for v in fs1 fs2 fs3; do
        models=""
        for m in xgbl xgb cat tabm; do
            if [ -f "preds/${m}_${v}.npz" ]; then models="$models ${m}_${v}"; fi
        done
        if [ -z "$models" ]; then continue; fi
        all="$all $models"
        run "blend_${v}" python blend.py $models --out "$v"
    done
    run blend_bag python blend.py $all --out bag
    log "summary of nested (honest) blend scores:"
    grep -H "nested" logs/blend_*.log || true
}

stages=${*:-prep select tune train blend}
for s in $stages; do
    log "=== stage $s ==="
    "stage_$s"
done
log "done. submissions: $(ls submission_fs*.csv submission_bag*.csv 2>/dev/null | tr '\n' ' ')"
