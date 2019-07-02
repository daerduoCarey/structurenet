import os
import sys
import random
from argparse import ArgumentParser
import numpy as np
import math
import torch
import torch.nn.functional as F
import utils
from config import add_eval_args
from data_partnetobb import PartNetObbDataset, Tree, collate_feats, part_name2id
from chamfer_distance import ChamferDistance
from geometry_utils import load_pts, export_ply_with_label, export_pts, export_label
from utils import linear_assignment
import provider
import compute_sym

sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

chamferLoss = ChamferDistance()

parser = ArgumentParser()
parser = add_eval_args(parser)
eval_conf = parser.parse_args()
conf = torch.load(os.path.join(eval_conf.model_path, eval_conf.name, 'conf.pth'))

# legacy
if 'max_tree_depth' not in conf:
    conf.max_tree_depth = 100

# configuration parameters that are not explicitly given as argument
# are copied from the training configuration
if len(eval_conf.data_path) == 0:
    eval_conf.data_path = conf.data_path
if len(eval_conf.data_type) == 0:
    eval_conf.data_type = conf.data_type

# merge training and evaluation configurations, giving evaluation parameters precendence
conf.__dict__.update(eval_conf.__dict__)

models = utils.get_model_module(version=conf.model_version)

chamfer_loss = ChamferDistance()

device = torch.device(conf.device)
print(f'Using device: {conf.device}')

# control randomness
if conf.seed < 0:
    conf.seed = random.randint(1, 10000)
print("Random Seed: %d" % (conf.seed))
random.seed(conf.seed)
np.random.seed(conf.seed)
torch.manual_seed(conf.seed)
if conf.deterministic and 'cuda' in conf.device:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if not os.path.exists(os.path.join(conf.result_path, conf.name)):
    os.makedirs(os.path.join(conf.result_path, conf.name))

# create models
encoder = models.RecursiveEncoder(conf, variational=True, discriminator=False, probabilistic=False, child_encoder_type=conf.child_encoder_type)
decoder = models.RecursiveDecoder(conf)

models = [encoder, decoder]
model_names = ['vae_encoder', 'vae_decoder']

__ = utils.load_checkpoint(
    models=models, model_names=model_names,
    dirname=os.path.join(conf.model_path, conf.name),
    epoch=conf.model_epoch if conf.model_epoch >= 0 else None,
    strict=True) # strict=False)

# create dataset
data_features = ['object', 'name']
dataset = PartNetObbDataset(root=conf.data_path, object_list=conf.dataset, data_features=data_features, load_geo=conf.load_geo)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_feats)

for m in models:
    m.to(device)

# set models to evaluation mode
for m in models:
    m.eval()

num_batch = len(dataloader)
with torch.no_grad():
    for batch_ind, batch in enumerate(dataloader):
        obj = batch[data_features.index('object')][0]
        obj.to(device)
        obj_name = batch[data_features.index('name')][0]

        root_code_and_kld = encoder.encode_structure(obj=obj)
        root_code = root_code_and_kld[:, :conf.feature_size]
        recon_obj = decoder.decode_structure(z=root_code, max_depth=conf.max_tree_depth)

        # save original and reconstructed object
        os.mkdir(os.path.join(conf.result_path, conf.name, obj_name))
        orig_output_filename = os.path.join(conf.result_path, conf.name, obj_name, 'orig.ply')
        recon_output_filename = os.path.join(conf.result_path, conf.name, obj_name, 'recon.ply')
        
        orig_geos, orig_nodes = obj.root.geos(leafs_only=True)
        orig_label = np.hstack([np.ones((conf.num_point), dtype=np.int32)*i for i in range(len(orig_geos))])
        orig_sem_label = np.hstack([np.ones((conf.num_point), dtype=np.int32)*part_name2id[n.full_label] for n in orig_nodes])
        orig_pc = torch.cat(orig_geos, dim=0).view(-1, 3).cpu().detach().numpy()
        export_ply_with_label(orig_output_filename, orig_pc, orig_label)
        export_pts(orig_output_filename.replace('ply', 'pts'), orig_pc)
        export_label(orig_output_filename.replace('.ply', '-ins.label'), orig_label)
        export_label(orig_output_filename.replace('.ply', '-sem.label'), orig_sem_label)

        recon_geos, recon_nodes = recon_obj.root.geos(leafs_only=True)
        recon_label = np.hstack([np.ones((conf.num_point), dtype=np.int32)*i for i in range(len(recon_geos))])
        recon_sem_label = np.hstack([np.ones((conf.num_point), dtype=np.int32)*part_name2id[n.full_label] for n in recon_nodes])
        recon_pc = torch.cat(recon_geos, dim=0).view(-1, 3).cpu().detach().numpy()
        export_ply_with_label(recon_output_filename, recon_pc, recon_label)
        export_pts(recon_output_filename.replace('ply', 'pts'), recon_pc)
        export_label(recon_output_filename.replace('.ply', '-ins.label'), recon_label)
        export_label(recon_output_filename.replace('.ply', '-sem.label'), recon_sem_label)


