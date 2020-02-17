# StructureNet Experiments
This folder includes the StructureNet experiments for AE reconstruction and VAE generation using both box-shape and point cloud representations. 

## Before start
To train the models, please first go to `data/partnetdata/` folder and download the training data. 
To test over the pretrained models, please go to `data/models/` folder and download the pretrained checkpoints.

## Dependencies
This code has been tested on Ubuntu 16.04 with Cuda 9.0, GCC 5.4.0, Python 3.6.5, PyTorch 1.1.0, Jupyter IPython Notebook 5.7.8. 

Please run
    
    pip3 install -r requirements.txt

to install the other dependencies.

Then, install https://github.com/rusty1s/pytorch_scatter by running

    pip3 install torch-scatter


## Box-shape AE Reconstruction
To train the network from scratch, run 

    bash scripts/train_box_ae_chair.sh

To test the model, run

    bash scripts/eval_recon_box_ae_chair.sh

After running this script, the evaluation statistics reported in Table 1 of the paper will be stored at `../data/results/box_ae_chair/stats.txt`.

You can use `vis_box.ipynb` to visualize the box-shape reconstruction results.

## Box-shape VAE Generation
To train the network from scratch, run

    bash scripts/train_box_vae_chair.sh

To test the model, run

    bash scripts/eval_gen_box_vae_chair.sh

You can use `vis_box.ipynb` to visualize the box-shape generation results.

## Point-cloud AE Reconstruction
First, pretrain Part-PC-AE using

    bash scripts/pretrain_part_pc_ae_chair.sh

Next, to train the network, run

    bash scripts/train_pc_ae_chair.sh

To test the model, run

    bash scripts/eval_recon_pc_ae_chair.sh

You can use `vis_pc.ipynb` to visualize the point cloud reconstruction results.

## Point-cloud VAE Generation
First, pretrain Part-PC-VAE using

    bash scripts/pretrain_part_pc_vae_chair.sh

Next, to train the network, run

    bash scripts/train_pc_vae_chair.sh

To test the model, run

    bash scripts/eval_gen_pc_vae_chair.sh

You can use `vis_pc.ipynb` to visualize the point cloud generation results.

## No-edge version (A simpler version)

If you are looking for a simpler starting point, please refer to [this](https://github.com/daerduoCarey/structedit/blob/master/code/model_structurenet.py) for a no-edge version of the network.
