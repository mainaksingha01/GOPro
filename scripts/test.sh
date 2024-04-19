cd ../..

# custom config
DATA=data
TRAINER=GOPro

DATASET=$1
TASK=$2
# SEED=$3

CFG=vit_b16
SHOTS=16
# LOADEP=50
SUB=new

for SEED in 1 2 3 4 5
do
    COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

    MODEL_DIR=GOPro/outputs/${TASK}/train_base/${COMMON_DIR}
    DIR=GOPro/outputs/${TASK}/test_${SUB}/${COMMON_DIR}
    if [ -d "$DIR" ]; then
        echo "The results already exist in ${DIR}"
    else
        python GOPro/train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file GOPro/configs/datasets/${DATASET}.yaml \
        --config-file GOPro/configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB} 
    fi
done

#--load-epoch ${LOADEP} \