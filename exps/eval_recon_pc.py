"""
    This is the main tester script for point cloud reconstruction evaluation.
    Use scripts/eval_recon_pc_ae_chair.sh to run.
"""

import os
import sys
import shutil
from argparse import ArgumentParser
import numpy as np
import torch
import utils
from config import add_eval_args
from data import PartNetDataset, Tree

sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

parser = ArgumentParser()
parser = add_eval_args(parser)
eval_conf = parser.parse_args()

# load train config
conf = torch.load(os.path.join(eval_conf.model_path, eval_conf.exp_name, 'conf.pth'))
eval_conf.data_path = conf.data_path

# load object category information
Tree.load_category_info(conf.category)

# merge training and evaluation configurations, giving evaluation parameters precendence
conf.__dict__.update(eval_conf.__dict__)

# load model
models = utils.get_model_module(conf.model_version)

# set up device
device = torch.device(conf.device)
print(f'Using device: {conf.device}')

# check if eval results already exist. If so, delete it. 
if os.path.exists(os.path.join(conf.result_path, conf.exp_name)):
    response = input('Eval results for "%s" already exists, overwrite? (y/n) ' % (conf.exp_name))
    if response != 'y':
        sys.exit()
    shutil.rmtree(os.path.join(conf.result_path, conf.exp_name))

# create a new directory to store eval results
os.makedirs(os.path.join(conf.result_path, conf.exp_name))

# create models
encoder = models.RecursiveEncoder(conf, variational=True, probabilistic=False)
decoder = models.RecursiveDecoder(conf)
models = [encoder, decoder]
model_names = ['encoder', 'decoder']

# load pretrained model
__ = utils.load_checkpoint(
    models=models, model_names=model_names,
    dirname=os.path.join(conf.model_path, conf.exp_name),
    epoch=conf.model_epoch,
    strict=True)

# create dataset
data_features = ['object', 'name']
dataset = PartNetDataset(conf.data_path, conf.test_dataset, data_features, load_geo=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_feats)

# send to device
for m in models:
    m.to(device)

# set models to evaluation mode
for m in models:
    m.eval()

# test over all test shapes
num_batch = len(dataloader)
with torch.no_grad():
    for batch_ind, batch in enumerate(dataloader):
        obj = batch[data_features.index('object')][0]
        obj.to(device)
        obj_name = batch[data_features.index('name')][0]

        root_code_and_kld = encoder.encode_structure(obj=obj)
        root_code = root_code_and_kld[:, :conf.feature_size]
        recon_obj = decoder.decode_structure(z=root_code, max_depth=conf.max_tree_depth)
        print('[%d/%d] ' % (batch_ind, num_batch), obj_name)

        # save original and reconstructed object
        os.mkdir(os.path.join(conf.result_path, conf.exp_name, obj_name))
        orig_output_filename = os.path.join(conf.result_path, conf.exp_name, obj_name, 'orig.json')
        recon_output_filename = os.path.join(conf.result_path, conf.exp_name, obj_name, 'recon.json')
        PartNetDataset.save_object(obj=obj, fn=orig_output_filename)
        PartNetDataset.save_object(obj=recon_obj, fn=recon_output_filename)

