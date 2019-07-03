import os
import sys
import random
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import utils
from config import add_eval_args
from geometry_utils import export_pts
import provider

sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

parser = ArgumentParser()
parser = add_eval_args(parser)
eval_conf = parser.parse_args()
conf = torch.load(os.path.join(eval_conf.model_path, eval_conf.name, 'conf.pth'))

class PartNetGeoDataset(torch.utils.data.Dataset):
    def __init__(self, root, object_list, use_local_frame):
        self.root = root
        self.use_local_frame = use_local_frame
        
        if isinstance(object_list, str):
            with open(object_list, 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]
        else:
            self.object_names = object_list

    def __getitem__(self, index):
        fn = os.path.join(self.root, self.object_names[index]+'.npz')
        data = np.load(fn)['parts']
        idx = np.random.randint(data.shape[0])
        pts = torch.tensor(data[idx, :, :], dtype=torch.float32)
        if self.use_local_frame:
            pts = pts - pts.mean(dim=0)
            pts = pts / pts.pow(2).sum(dim=1).max().sqrt()
        return (pts, self.object_names[index], idx)

    def __len__(self):
        return len(self.object_names)

def collate_feats(b):
    return list(zip(*b))

# configuration parameters that are not explicitly given as argument
# are copied from the training configuration
if len(eval_conf.data_path) == 0:
    eval_conf.data_path = conf.data_path
if len(eval_conf.data_type) == 0:
    eval_conf.data_type = conf.data_type

# merge training and evaluation configurations, giving evaluation parameters precendence
conf.__dict__.update(eval_conf.__dict__)

models = utils.get_model_module(version=conf.model_version)

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
encoder = models.PartEncoder(feat_len=100, probabilistic=False)
decoder = models.PartDecoder(feat_len=100, num_point=1000)

models = [encoder, decoder]
model_names = ['part_vae_encoder', 'part_vae_decoder']

__ = utils.load_checkpoint(
    models=models, model_names=model_names,
    dirname=os.path.join(conf.model_path, conf.name),
    epoch=conf.model_epoch if conf.model_epoch >= 0 else None,
    strict=True) # strict=False)

# create dataset
dataset = PartNetGeoDataset(root=conf.data_path, object_list=conf.dataset, use_local_frame=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_feats)

for m in models:
    m.to(device)

# set models to evaluation mode
for m in models:
    m.eval()

with torch.no_grad():
    for batch_ind, batch in enumerate(dataloader):
        pts = torch.cat([item.unsqueeze(dim=0) for item in batch[0]], dim=0).to(device)
        print('pts: ', pts.size())
        net = encoder(pts)
        print('net: ', net.size())
        pred = decoder(net)
        print('pred: ', pred.size())
        pts = pts.cpu().numpy()
        pred = pred.cpu().numpy()
        for i in range(32):
            export_pts(os.path.join(conf.result_path, conf.name, 'orig-%02d.pts'%i), pts[i])
            export_pts(os.path.join(conf.result_path, conf.name, 'pred-%02d.pts'%i), pred[i])
        break

