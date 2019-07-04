python ./train_box.py \
  --exp_name 'box_ae_chair' \
  --category 'Chair' \
  --data_path '../data/partnetdata/chair_hier' \
  --train_dataset 'train_no_other_less_than_10_parts.txt' \
  --val_dataset 'val_no_other_less_than_10_parts.txt' \
  --epochs 200 \
  --model_version 'model_box' \
  --non_variational
