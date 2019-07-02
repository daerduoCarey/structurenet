# This file contains the training script for AE experiment for reconstruction for box-represented shapes

python ./train_box.py \
  --exp_name 'box_ae' \
  --data_path '../data/partnetdata_chair' \
  --train_dataset '../data/partnetdata_chair/train.txt' \
  --val_dataset '../data/partnetdata_chair/val.txt' \
  --epochs 200 \
  --model_version 'model_box' \
  --num_gnn_iterations 2 \
  --num_dec_gnn_iterations 2 \
  --symmetric_type 'max' \
  --edge_symmetric_type 'avg' \
  --dec_edge_symmetric_type 'avg' \
  --loss_weight_sym 1.0 \
  --loss_weight_adj 1.0 \
  --loss_weight_box 20 \
  --loss_weight_anchor 10 \
  --non_variational

