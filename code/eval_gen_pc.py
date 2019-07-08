"""
    This is the main tester script for point cloud generation evaluation.
    Use scripts/eval_gen_pc_vae_chair.sh to run.
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
parser.add_argument('--num_gen', type=int, default=1, help='how many shapes to generate?')
eval_conf = parser.parse_args()

# load train config
conf = torch.load(os.path.join(eval_conf.model_path, eval_conf.exp_name, 'conf.pth'))

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
decoder = models.RecursiveDecoder(conf)
models = [decoder]
model_names = ['decoder']

# load pretrained model
__ = utils.load_checkpoint(
    models=models, model_names=model_names,
    dirname=os.path.join(conf.model_path, conf.exp_name),
    epoch=conf.model_epoch,
    strict=True)

# send to device
for m in models:
    m.to(device)

# set models to evaluation mode
for m in models:
    m.eval()

# generate shapes
with torch.no_grad():
    for i in range(conf.num_gen):
        print(f'Generating {i}/{conf.num_gen} ...')
        code = torch.randn(1, conf.feature_size).cuda()
        obj = decoder.decode_structure(z=code, max_depth=conf.max_tree_depth)
        output_filename = os.path.join(conf.result_path, conf.exp_name, 'object-%04d.json'%i)
        PartNetDataset.save_object(obj=obj, fn=output_filename)

