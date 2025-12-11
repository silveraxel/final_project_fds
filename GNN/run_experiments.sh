#!/bin/bash

# Create output directory for models and logs
OUTPUT_DIR="./experiments"
LOG_DIR="${OUTPUT_DIR}/logs"
PLOT_DIR="${OUTPUT_DIR}/plots"
MODELS_DIR="${OUTPUT_DIR}/models"

mkdir -p "${LOG_DIR}"
mkdir -p "${MODELS_DIR}"
mkdir -p "${PLOT_DIR}"

# ============================================================================
# PARAMETER DEFINITIONS
# ============================================================================

# Define parameter arrays to iterate over
LEARNING_RATES=(0.001 0.005 0.01)
LOSS_TYPES=("L2")
NUM_GNN_LAYERS=(3 4)
NUM_MLP_LAYERS=(1 2 3)
BATCH_SIZES=(512 256)
TAG_AS_EDGE_VALUES=(true)
GRADIENT_CLIPPING_VALUES=(true)

#Common parameters

COMMON_PARAMS="--dropout 0.3  --early-stopping-patience 50"

declare -A NEIGHBOR_CONFIGS

# For 3 layers
NEIGHBOR_CONFIGS["3_aggressive"]="30 20 10"
NEIGHBOR_CONFIGS["3_conservative"]="15 10 5"

# For 4 layers
NEIGHBOR_CONFIGS["4_aggressive"]="30 20 10 5"
NEIGHBOR_CONFIGS["4_conservative"]="15 10 5 1"

SAMPLING_STRATEGIES=("aggressive" "conservative")

echo "========================================"
echo "Starting Experiment Suite"
echo "========================================"
echo ""

# ============================================================================
# MAIN ITERATION LOOP
# ============================================================================


EXPERIMENT_COUNT=0

for lr in "${LEARNING_RATES[@]}"; do
    for loss in "${LOSS_TYPES[@]}"; do
        for gnn_layers in "${NUM_GNN_LAYERS[@]}"; do
            for mlp_layers in "${NUM_MLP_LAYERS[@]}"; do
                for strategy in "${SAMPLING_STRATEGIES[@]}"; do
                    for batch_size in "${BATCH_SIZES[@]}"; do
                        for tag_as_edge in "${TAG_AS_EDGE_VALUES[@]}"; do
                            for grad_clip in "${GRADIENT_CLIPPING_VALUES[@]}"; do
                                
                                EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
            
                                MODEL_NAME="model_${EXPERIMENT_COUNT}_${loss}"
                                MODEL_PATH="${MODELS_DIR}/${MODEL_NAME}.pt"
                                TRAIN_LOG="${LOG_DIR}/${MODEL_NAME}_train.log"
                                INFERENCE_LOG="${LOG_DIR}/${MODEL_NAME}_inference.log"                                
                                
                                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training..."
                                
                                config_key="${gnn_layers}_${strategy}"
                                neighbors="${NEIGHBOR_CONFIGS[$config_key]}"

                                TRAINING_PARAMS="--modality training"
                                TRAINING_PARAMS="${TRAINING_PARAMS} --model_path ${MODEL_PATH}"
                                TRAINING_PARAMS="${TRAINING_PARAMS} --learning-rate ${lr}"
                                TRAINING_PARAMS="${TRAINING_PARAMS} --loss_type ${loss}"
                                TRAINING_PARAMS="${TRAINING_PARAMS} --num-gnn-layers ${gnn_layers}"
                                TRAINING_PARAMS="${TRAINING_PARAMS} --num-mlp-layers ${mlp_layers}"
                                TRAINING_PARAMS="${TRAINING_PARAMS} --num-neighbors ${neighbors}"
                                TRAINING_PARAMS="${TRAINING_PARAMS} --batch-size ${batch_size}"

                                if [ "$tag_as_edge" = true ]; then
                                    TRAINING_PARAMS="${TRAINING_PARAMS} --tag-as-edge"
                                else
                                    TRAINING_PARAMS="${TRAINING_PARAMS} --no-tag-as-edge"
                                fi
                                
                                if [ "$grad_clip" = true ]; then
                                    TRAINING_PARAMS="${TRAINING_PARAMS} --use-gradient-clipping"
                                else
                                    TRAINING_PARAMS="${TRAINING_PARAMS} --no-use-gradient-clipping"
                                fi

                                echo "Using the following training params"
                                echo "${TRAINING_PARAMS} ${COMMON_PARAMS}"

                                python3 main.py ${TRAINING_PARAMS} ${COMMON_PARAMS}
                                
                                if [ -f "${MODEL_NAME}_loss.png" ]; then
                                    mv "${MODEL_NAME}_loss.png" "${PLOT_DIR}/"
                                fi

                                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed successfully"
                                echo ""
                                
                                # --------------------------------------------------------------------
                                # INFERENCE PHASE
                                # --------------------------------------------------------------------

                                INFERENCE_PARAMS="--modality inference"
                                INFERENCE_PARAMS="${INFERENCE_PARAMS} --model_path ${MODEL_PATH}"

                                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting inference..."
                                
                                python3 main.py  ${INFERENCE_PARAMS} 
    
                                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Inference completed successfully"

                                #Moving files

                                if [ -f "${MODEL_NAME}_analysis.png" ]; then
                                    mv "${MODEL_NAME}_analysis.png" "${PLOT_DIR}/"
                                fi

                                if [ -f "${MODEL_NAME}.pt" ]; then
                                    mv "${MODEL_NAME}.pt" "${MODELS_DIR}/"
                                fi

                                if [ -f "${MODEL_NAME}_parameters.txt" ]; then
                                    mv "${MODEL_NAME}_parameters.txt" "${MODELS_DIR}/"
                                fi

                                if [ -f "${MODEL_NAME}_params_data.json" ]; then
                                    mv "${MODEL_NAME}_params_data.json" "${MODELS_DIR}/"
                                fi

                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# ============================================================================
# SUMMARY
# ============================================================================
echo "========================================"
echo "Experiment Suite Completed"
echo "========================================"
echo "Total experiments run: ${EXPERIMENT_COUNT}"
echo "Results saved in: ${OUTPUT_DIR}"

exit 0
