# StructureNet Experiments
This folder includes the StructureNet experiments for AE reconstruction and VAE generation using both box-shape and point cloud representations. 

## Before start
To train the models, please first go to `data/partnetdata/` folder and download the training data. 
To test over the pretrained models, please go to `data/models/` folder and download the pretrained checkpoints.

## Box-shape AE Reconstruction
To train the network from scratch, run 

    bash scripts/train_box_ae_chair.sh

To test the pretrained model, run

    bash scripts/eval_recon_box_ae_chair.sh

The evaluation statistics reported in Table 1 of the paper is stored at `../data/results/box_ae_chair/stats.txt`.

You can use `vis_box.ipynb` to visualize the box-shape reconstruction results.



