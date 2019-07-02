set -ex
# name from filename
NAME=$0
NAME=${NAME##*/}
NAME=${NAME%.*}
NAME=${NAME#eval_}

DATASET='../data/partnetobb_chair/test_no_other_less_than_10_parts.txt'
#DATASET='../data/partnetobb_chair/val_2266.txt'
MODEL_EPOCH=101

python eval_vae_v2.py \
  --name ${NAME} \
  --dataset ${DATASET} \
  --model_epoch ${MODEL_EPOCH}
