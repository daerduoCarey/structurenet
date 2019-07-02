set -ex
# name from filename
NAME=$0
NAME=${NAME##*/}
NAME=${NAME%.*}
NAME=${NAME#eval_}

DATASET='../data/partnetobb_chair/val_no_other_less_than_10_parts.txt'
#DATASET='../data/partnetobb_chair/val_2266.txt'
MODEL_EPOCH=162

python eval_vae_v2_pc.py \
  --name ${NAME} \
  --dataset ${DATASET} \
  --model_epoch ${MODEL_EPOCH}
