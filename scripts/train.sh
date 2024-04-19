cd ../..

# custom config
DATA=data 
TRAINER=GOPro

DATASET=$1
TASK=$2  # base2new, cross-dataset, domain-generalization
# SEED=$3

CFG=vit_b16
SUB=base  # base, new, all
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)

for SEED in 1 2 3 4 5
do 
    DIR=GOPro/outputs/${TASK}/train_${SUB}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
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
        TRAINER.GOPRO.N_CTX ${NCTX} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
