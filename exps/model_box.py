import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from chamfer_distance import ChamferDistance
from data_partnetobb import Tree, part_name2cids, part_id2name, num_sem, part_non_leaf_sem_names
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from geometry_utils import load_pts, export_pts, export_ply_with_label
import provider
from utils import linear_assignment
import torch_scatter
import compute_sym

# with edge encoding/decoding

#########################################################################################
## Encoder
#########################################################################################

class Sampler(nn.Module):

    def __init__(self, feature_size, hidden_size, probabilistic=True):
        super(Sampler, self).__init__()

        self.probabilistic = probabilistic

        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, feature_size)

        # if probabilistic:
        self.mlp2var = nn.Linear(hidden_size, feature_size)

    def forward(self, x):
        encode = torch.relu(self.mlp1(x))
        mu = self.mlp2mu(encode)

        if self.probabilistic:
            logvar = self.mlp2var(encode)
            # log(variance) to standard deviation
            std = logvar.mul(0.5).exp_()
            # standard normal distributed noise
            eps = torch.randn_like(std)

            # KL-divergence
            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)

            return torch.cat([eps.mul(std).add_(mu), kld], 1)
        else:
            return mu


class BoxEncoder(nn.Module):

    def __init__(self, feature_size):
        super(BoxEncoder, self).__init__()
        self.encoder = nn.Linear(10, feature_size)

    def forward(self, box_input):
        box_vector = torch.relu(self.encoder(box_input))
        return box_vector


# order invariant
class SymmetricChildEncoder(nn.Module):

    def __init__(self, feature_size, hidden_size, symmetric_type):
        super(SymmetricChildEncoder, self).__init__()

        self.child_op = nn.Linear(feature_size+num_sem, hidden_size)
        self.second = nn.Linear(hidden_size, feature_size)

        print(f'Using Symmetric Child Encoder with Symmetric Type: {symmetric_type}')
        self.symmetric_type = symmetric_type
        print(feature_size, hidden_size)

    # child feats: b x max_childs x feats
    # child exists: b x max_childs x 1
    def forward(self, child_feats, child_exists):

        batch_size = child_feats.shape[0]
        max_childs = child_feats.shape[1]

        child_feats = torch.relu(self.child_op(child_feats))

        # zero non-existent children
        child_feats = child_feats * child_exists

        child_feats = child_feats.view(batch_size, max_childs, -1)

        if self.symmetric_type == 'max':
            parent_feat = child_feats.max(dim=1)[0]
        elif self.symmetric_type == 'sum':
            parent_feat = child_feats.sum(dim=1)
        elif self.symmetric_type == 'avg':
            parent_feat = child_feats.sum(dim=1) / child_exists.sum(dim=1)
        else:
            raise ValueError(f'Unknown symmetric type: {self.symmetric_type}')

        parent_feat = torch.relu(self.second(parent_feat))

        return parent_feat


class GNNChildEncoder(nn.Module):

    def __init__(self, node_feat_size, node_hidden_size, edge_feat_size, symmetric_type, edge_symmetric_type, num_iterations, shared_iterations, edge_type_num):
        super(GNNChildEncoder, self).__init__()
        self.symmetric_type = symmetric_type
        self.edge_symmetric_type = edge_symmetric_type
        self.num_iterations = num_iterations
        self.shared_iterations = shared_iterations
        self.edge_type_num = edge_type_num

        self.child_op = nn.Linear(node_feat_size+num_sem, node_hidden_size)
        if shared_iterations:
            self.node_edge_op = nn.Linear(node_hidden_size*2+edge_feat_size+edge_type_num, node_hidden_size)
        else:
            self.node_edge_op = torch.nn.ModuleList()
            for i in range(self.num_iterations):
                self.node_edge_op.append(nn.Linear(node_hidden_size*2+edge_feat_size+edge_type_num, node_hidden_size))

        self.parent_op = nn.Linear(node_hidden_size*(self.num_iterations+1), node_feat_size)

        print(f'''GNN Child Encoder [symmetric type: {symmetric_type}, edge symmetric type: {edge_symmetric_type}, '''
              f'''node feat. size (hidden): {node_feat_size} ({node_hidden_size}), '''
              f'''edge feat. size: {edge_feat_size}, '''
              f'''num. iterations: {num_iterations}]''')

    # child feats: b x max_childs x feat_dim
    # child exists: b x max_childs x 1
    # edge_type_onehot: b x num_edges x edge_type_num
    # edge_feats: b x num_edges x edge_feat_dim
    # edge_indices: b x num_edges x 2
    def forward(self, child_feats, child_exists, edge_type_onehot, edge_feats, edge_indices):

        batch_size = child_feats.shape[0]
        max_childs = child_feats.shape[1]
        num_edges = edge_indices.shape[1]

        if batch_size != 1:
            raise ValueError('Currently only a single batch is supported.')
            # would need to do indexing with batched edge indices
            # and possibly different number of edges in different batches
            # could be done with sparse adjacency matrix for edge_indices,
            # but I would need to make sure edge_feats has the correct order
            # or would also need to make edge_feats sparse
            # (although sparse only in some dimensions)

        # MLP for child features
        child_feats = torch.relu(self.child_op(child_feats))
        node_hidden_size = child_feats.size(-1)

        # zero non-existent children
        child_feats = child_feats * child_exists
        child_feats = child_feats.view(1, max_childs, -1)

        # combine node features into a graph feature
        iter_parent_feats = []
        if self.symmetric_type == 'max':
            iter_parent_feats.append(child_feats.max(dim=1)[0])
        elif self.symmetric_type == 'sum':
            iter_parent_feats.append(child_feats.sum(dim=1))
        elif self.symmetric_type == 'avg':
            iter_parent_feats.append(child_feats.sum(dim=1) / child_exists.sum(dim=1))
        else:
            raise ValueError(f'Unknown symmetric type: {self.symmetric_type}')

        if self.num_iterations > 0 and num_edges > 0:
            # MLP for edge features
            if edge_feats is not None:
                edge_feats = torch.cat([edge_type_onehot, edge_feats], dim=2)
            else:
                edge_feats = edge_type_onehot

        edge_indices_from = edge_indices[:, :, 0].view(-1, 1).expand(-1, node_hidden_size)

        # GNN
        # similar to dynamic graph CNN (https://arxiv.org/pdf/1801.07829.pdf)
        # and GIN (https://arxiv.org/pdf/1810.00826.pdf),
        # but with edge features
        for i in range(self.num_iterations):

            if num_edges > 0:
                # MLP for edge features concatenated with adjacent node features
                node_edge_feats = torch.cat([
                    child_feats[0:1, edge_indices[0, :, 0], :], # start node features
                    child_feats[0:1, edge_indices[0, :, 1], :], # end node features
                    edge_feats], dim=2) # edge features

                node_edge_feats = node_edge_feats.view(num_edges, -1)
                if self.shared_iterations:
                    node_edge_feats = torch.relu(self.node_edge_op(node_edge_feats))
                else:
                    node_edge_feats = torch.relu(self.node_edge_op[i](node_edge_feats))
                node_edge_feats = node_edge_feats.view(num_edges, -1)

                # aggregate information from neighboring nodes
                new_child_feats = child_feats.new_zeros(max_childs, node_hidden_size)
                if self.edge_symmetric_type == 'max':
                    new_child_feats, _ = torch_scatter.scatter_max(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                elif self.edge_symmetric_type == 'sum':
                    new_child_feats = torch_scatter.scatter_add(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                elif self.edge_symmetric_type == 'avg':
                    new_child_feats = torch_scatter.scatter_mean(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                else:
                    raise ValueError(f'Unknown edge symmetric type: {self.edge_symmetric_type}')

                child_feats = new_child_feats.view(1, max_childs, node_hidden_size)

            # combine node features into a graph feature
            if self.symmetric_type == 'max':
                iter_parent_feats.append(child_feats.max(dim=1)[0])
            elif self.symmetric_type == 'sum':
                iter_parent_feats.append(child_feats.sum(dim=1))
            elif self.symmetric_type == 'avg':
                iter_parent_feats.append(child_feats.sum(dim=1) / child_exists.sum(dim=1))
            else:
                raise ValueError(f'Unknown symmetric type: {self.symmetric_type}')

        # concatenation of the parent features from all iterations (as in GIN, like skip connections)
        parent_feat = torch.cat(iter_parent_feats, dim=1)

        # back to standard feature space size
        parent_feat = torch.relu(self.parent_op(parent_feat))

        return parent_feat


class RecursiveEncoder(nn.Module):

    def __init__(self, config, variational=False, discriminator=False, probabilistic=True, child_encoder_type='concat'):
        super(RecursiveEncoder, self).__init__()

        self.conf = config
        self.child_encoder_type = child_encoder_type

        self.box_encoder = BoxEncoder(feature_size=config.feature_size)
        if child_encoder_type == 'concat':
            self.child_encoder = ConcatChildEncoder(feature_size=config.feature_size, hidden_size=config.hidden_size, max_child_num=config.max_child_num)
        elif child_encoder_type == 'sym':
            self.child_encoder = SymmetricChildEncoder(feature_size=config.feature_size, hidden_size=config.hidden_size, symmetric_type=config.symmetric_type)
        elif child_encoder_type == 'gnn':
            self.child_encoder = GNNChildEncoder(
                node_feat_size=config.feature_size,
                node_hidden_size=config.hidden_size,
                edge_feat_size=config.edge_feature_size,
                symmetric_type=config.symmetric_type,
                edge_symmetric_type=config.edge_symmetric_type,
                num_iterations=config.num_gnn_iterations,
                shared_iterations=config.shared_gnn_iterations,
                edge_type_num=len(config.edge_types))
        else:
            raise ValueError(f'Unknown child encoder type: {child_encoder_type}')

        if variational:
            self.sample_encoder = Sampler(feature_size=config.feature_size, hidden_size=config.hidden_size, probabilistic=probabilistic)

    def boxEncoder(self, box):
        return self.box_encoder(box)

    def childEncoder(self, child_feats, child_exists, edge_type_onehot=None, edge_feats=None, edge_indices=None):
        if 'gnn' in self.child_encoder_type:
            return self.child_encoder(child_feats, child_exists, edge_type_onehot, edge_feats, edge_indices)
        else:
            return self.child_encoder(child_feats, child_exists)

    def sampleEncoder(self, feature):
        return self.sample_encoder(feature)

    def encode_node(self, node, fold=None, interp_sem=None, interp_part_code=None):
        if node.full_label == interp_sem:
            return interp_part_code

        if node.is_leaf:
            return self.boxEncoder(node.get_box_quat().view(1, -1))
        else:
            # get features of all children
            child_feats = []
            for child in node.children:
                cur_child_feat = torch.cat([self.encode_node(node=child, fold=fold, interp_sem=interp_sem, interp_part_code=interp_part_code), child.get_semantic_one_hot()], dim=1)
                child_feats.append(cur_child_feat.unsqueeze(dim=1))
            child_feats = torch.cat(child_feats, dim=1)

            if child_feats.shape[1] > self.conf.max_child_num:
                raise ValueError('Node has too many children.')

            # pad with zeros
            if child_feats.shape[1] < self.conf.max_child_num:
                padding = child_feats.new_zeros(child_feats.shape[0], self.conf.max_child_num-child_feats.shape[1], child_feats.shape[2])
                child_feats = torch.cat([child_feats, padding], dim=1)

            # 1 if the child exists, 0 if it is padded
            child_exists = child_feats.new_zeros(child_feats.shape[0], self.conf.max_child_num, 1)
            child_exists[:, :len(node.children), :] = 1

            if 'gnn' in self.child_encoder_type:
                # get feature of current node (parent of the children)
                edge_type_onehot, edge_feats, edge_indices = node.edge_tensors(
                    feat_type=self.conf.edge_feat_type, feat_size=self.conf.edge_feature_size,
                    edge_types=self.conf.edge_types, device=child_feats.device, type_onehot=True)

                return self.childEncoder(
                    child_feats=child_feats, child_exists=child_exists,
                    edge_type_onehot=edge_type_onehot, edge_feats=edge_feats, edge_indices=edge_indices)
            else:
                return self.childEncoder(child_feats=child_feats, child_exists=child_exists)

    def encode_structure(self, obj, fold=None, eval_time=False, interp_sem=None, interp_part_code=None):
        root_latent = self.encode_node(node=obj.root, fold=fold, interp_sem=interp_sem, interp_part_code=interp_part_code)
        return self.sampleEncoder(root_latent)


#########################################################################################
## Decoder
#########################################################################################

class LeafClassifier(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(LeafClassifier, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, 1)

    def forward(self, input_feature):
        output = torch.relu(self.mlp1(input_feature))
        output = self.mlp2(output)
        return output


class SampleDecoder(nn.Module):
    """ Decode a randomly sampled noise into a feature vector """
    def __init__(self, feature_size, hidden_size):
        super(SampleDecoder, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)

    def forward(self, input_feature):
        output = torch.relu(self.mlp1(input_feature))
        output = torch.relu(self.mlp2(output))
        return output


class BoxDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(BoxDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, hidden_size)
        self.center = nn.Linear(hidden_size, 3)
        self.size = nn.Linear(hidden_size, 3)
        self.quat = nn.Linear(hidden_size, 4)

    def forward(self, parent_feature):
        feat = torch.relu(self.mlp(parent_feature))
        center = torch.tanh(self.center(feat))
        size = torch.sigmoid(self.size(feat)) * 2
        quat_bias = feat.new_tensor([[1.0, 0.0, 0.0, 0.0]])
        quat = self.quat(feat).add(quat_bias)
        quat = quat / (1e-12 + quat.pow(2).sum(dim=1).unsqueeze(dim=1).sqrt())
        vector = torch.cat([center, size, quat], dim=1)
        return vector


class GNNChildDecoder(nn.Module):
    def __init__(self, node_feat_size, node_hidden_size, edge_feat_size, edge_dec_hidden_size, max_child_num, edge_symmetric_type, num_iterations, shared_iterations, edge_type_num):
        super(GNNChildDecoder, self).__init__()

        self.max_child_num = max_child_num
        self.node_hidden_size = node_hidden_size
        self.edge_dec_hidden_size = edge_dec_hidden_size
        self.edge_feat_size = edge_feat_size
        self.edge_symmetric_type = edge_symmetric_type
        self.num_iterations = num_iterations
        self.shared_iterations = shared_iterations
        self.edge_type_num = edge_type_num

        self.mlp_parent = nn.Linear(node_feat_size, node_hidden_size*max_child_num)
        self.mlp_exists = nn.Linear(node_hidden_size, 1)
        self.mlp_sem = nn.Linear(node_hidden_size, num_sem)
        self.mlp_child = nn.Linear(node_hidden_size, node_feat_size)
        self.mlp_edge_latent = nn.Linear(node_hidden_size*2, edge_dec_hidden_size)

        self.mlp_edge_exists = nn.ModuleList()
        for i in range(edge_type_num):
            self.mlp_edge_exists.append(nn.Linear(edge_dec_hidden_size, 1))

        if shared_iterations:
            self.node_edge_op = nn.Linear(node_hidden_size*2+edge_dec_hidden_size+edge_type_num, node_hidden_size)
        else:
            self.node_edge_op = torch.nn.ModuleList()
            for i in range(self.num_iterations):
                self.node_edge_op.append(nn.Linear(node_hidden_size*2+edge_dec_hidden_size+edge_type_num, node_hidden_size))

        self.mlp_child = nn.Linear(node_hidden_size*(self.num_iterations+1), node_hidden_size)
        self.mlp_sem = nn.Linear(node_hidden_size, num_sem)
        self.mlp_child2 = nn.Linear(node_hidden_size, node_feat_size)

        if self.edge_feat_size > 0:
            self.mlp_edge_feat = torch.nn.ModuleList()
            for i in range(edge_type_num):
                self.mlp_edge_feat.append(nn.Linear(edge_dec_hidden_size, edge_feat_size))

    def forward(self, parent_feature):

        batch_size = parent_feature.shape[0]
        feat_size = parent_feature.shape[1]

        if batch_size != 1:
            raise ValueError('Only batch size 1 supported for now.')

        parent_feature = torch.relu(self.mlp_parent(parent_feature))
        child_feats = parent_feature.view(batch_size, self.max_child_num, self.node_hidden_size)

        # node existence
        child_exists_logits = self.mlp_exists(child_feats.view(batch_size*self.max_child_num, self.node_hidden_size))
        child_exists_logits = child_exists_logits.view(batch_size, self.max_child_num, 1)

        # edge features
        edge_latents = torch.cat([
            child_feats.view(batch_size, self.max_child_num, 1, feat_size).expand(-1, -1, self.max_child_num, -1),
            child_feats.view(batch_size, 1, self.max_child_num, feat_size).expand(-1, self.max_child_num, -1, -1)
            ], dim=3)
        edge_latents = torch.relu(self.mlp_edge_latent(edge_latents))

        edge_exists_logits_per_type = []
        for i in range(self.edge_type_num):
            edge_exists_logits_cur_type = self.mlp_edge_exists[i](edge_latents).view(batch_size, self.max_child_num, self.max_child_num, 1)
            edge_exists_logits_per_type.append(edge_exists_logits_cur_type)
        edge_exists_logits = torch.cat(edge_exists_logits_per_type, dim=3)

        # message passing
        # there are several possible versions, this is a simple one:
        # use a fixed set of edges, consisting of existing edges connecting existing nodes
        # this set of edges does not change during iterations
        # iteratively update the child latent features
        # then use these child latent features to compute child features and semantics
        #
        # additional possibilities not implemented here:
        # - update child existence after the iterations or during iterations
        # - also update edge features
        # - update edge existence after the iterations or during iterations
        # - for a smaller model: don't use edge latent, only edge type

        # get edges that exist between nodes that exist
        edge_indices = torch.nonzero(edge_exists_logits > 0)
        edge_types = edge_indices[:, 3]
        edge_indices = edge_indices[:, 1:3]
        nodes_exist_mask = (child_exists_logits[0, edge_indices[:, 0], 0] > 0) & (child_exists_logits[0, edge_indices[:, 1], 0] > 0)
        edge_indices = edge_indices[nodes_exist_mask, :]
        edge_types = edge_types[nodes_exist_mask]

        # get latent features for the edges
        edge_feats_mp = edge_latents[0:1, edge_indices[:, 0], edge_indices[:, 1], :]

        # append edge type to edge features, so the network has information which
        # of the possibly multiple edges between two nodes it is working with
        edge_type_logit = edge_exists_logits[0:1, edge_indices[:, 0], edge_indices[:, 1], :]
        edge_type_logit = edge_feats_mp.new_zeros(edge_feats_mp.shape[:2]+(self.edge_type_num,))
        edge_type_logit[0:1, range(edge_type_logit.shape[1]), edge_types] = edge_exists_logits[0:1, edge_indices[:, 0], edge_indices[:, 1], edge_types]
        edge_feats_mp = torch.cat([edge_feats_mp, edge_type_logit], dim=2)

        num_edges = edge_indices.shape[0]
        max_childs = child_feats.shape[1]

        iter_child_feats = [child_feats] # zeroth iteration

        if self.num_iterations > 0 and num_edges > 0:
            edge_indices_from = edge_indices[:, 0].view(-1, 1).expand(-1, self.node_hidden_size)

        for i in range(self.num_iterations):
            if num_edges > 0:

                node_edge_feats = torch.cat([
                    child_feats[0:1, edge_indices[:, 0], :], # start node features
                    child_feats[0:1, edge_indices[:, 1], :], # end node features
                    edge_feats_mp], dim=2) # edge features

                node_edge_feats = node_edge_feats.view(num_edges, -1)
                if self.shared_iterations:
                    node_edge_feats = torch.relu(self.node_edge_op(node_edge_feats))
                else:
                    node_edge_feats = torch.relu(self.node_edge_op[i](node_edge_feats))

                # aggregate information from neighboring nodes
                new_child_feats = child_feats.new_zeros(max_childs, self.node_hidden_size)
                if self.edge_symmetric_type == 'max':
                    new_child_feats, _ = torch_scatter.scatter_max(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                elif self.edge_symmetric_type == 'sum':
                    new_child_feats = torch_scatter.scatter_add(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                elif self.edge_symmetric_type == 'avg':
                    new_child_feats = torch_scatter.scatter_mean(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                else:
                    raise ValueError(f'Unknown edge symmetric type: {self.edge_symmetric_type}')

                child_feats = new_child_feats.view(1, max_childs, self.node_hidden_size)

            # save child features of this iteration
            iter_child_feats.append(child_feats)

        # concatenation of the child features from all iterations (as in GIN, like skip connections)
        child_feats = torch.cat(iter_child_feats, dim=2)

        # transform concatenation back to original feature space size
        child_feats = child_feats.view(batch_size * self.max_child_num, self.node_hidden_size*(self.num_iterations+1))
        child_feats = torch.relu(self.mlp_child(child_feats))
        child_feats = child_feats.view(batch_size, self.max_child_num, self.node_hidden_size)

        # node semantics
        child_sem_logits = self.mlp_sem(child_feats.view(batch_size*self.max_child_num, self.node_hidden_size))
        child_sem_logits = child_sem_logits.view(batch_size, self.max_child_num, num_sem)

        # node features
        child_feats = self.mlp_child2(child_feats.view(batch_size*self.max_child_num, self.node_hidden_size))
        child_feats = child_feats.view(batch_size, self.max_child_num, feat_size)
        child_feats = torch.relu(child_feats)

        # edge features
        if self.edge_feat_size > 0:
            edge_feats_per_type = []
            for i in range(self.edge_type_num):
                edge_feats_cur_type = self.mlp_edge_feat[i](edge_latents).view(batch_size, self.max_child_num, self.max_child_num, 1, self.edge_feat_size)
                edge_feats_per_type.append(edge_feats_cur_type)
            edge_feats = torch.cat(edge_feats_per_type, dim=3)
        else:
            edge_feats = None

        return child_feats, child_sem_logits, child_exists_logits, edge_exists_logits, edge_feats


class RecursiveDecoder(nn.Module):
    def __init__(self, config, child_decoder_type='gnn'):
        super(RecursiveDecoder, self).__init__()

        self.conf = config

        self.box_decoder = BoxDecoder(feature_size=config.feature_size, hidden_size=config.hidden_size)

        if child_decoder_type == 'gnn':
            self.child_decoder = GNNChildDecoder(
                node_feat_size=config.feature_size,
                node_hidden_size=config.hidden_size,
                edge_feat_size=config.edge_feature_size,
                edge_dec_hidden_size=config.edge_dec_hidden_size,
                max_child_num=config.max_child_num,
                edge_symmetric_type=config.dec_edge_symmetric_type,
                num_iterations=config.num_dec_gnn_iterations,
                shared_iterations=config.shared_dec_gnn_iterations,
                edge_type_num=len(config.edge_types))
        else:
            raise ValueError(f'Unknown child decoder type: {child_decoder_type}')

        self.sample_decoder = SampleDecoder(feature_size=config.feature_size, hidden_size=config.hidden_size)
        self.leaf_classifier = LeafClassifier(feature_size=config.feature_size, hidden_size=config.hidden_size)
        self.bceLoss = nn.BCEWithLogitsLoss(reduction='none')
        self.chamferLoss = ChamferDistance()
        self.semCELoss = nn.CrossEntropyLoss(reduction='none')
        self.child_decoder_type = child_decoder_type

        self.register_buffer('unit_cube', torch.from_numpy(load_pts('cube.pts')))
        self.register_buffer('anchor', torch.from_numpy(load_pts('anchor.pts')))

    def boxDecoder(self, feature):
        return self.box_decoder(feature)

    def childDecoder(self, feature):
        return self.child_decoder(feature)

    def sampleDecoder(self, feature):
        return self.sample_decoder(feature)

    def leafClassifier(self, feature):
        return self.leaf_classifier(feature)

    def boxLossEstimator(self, box_feature, gt_box_feature):
        pred_box_pc = provider.transform_pc_batch(self.unit_cube, box_feature)
        with torch.no_grad():
            pred_reweight = provider.get_surface_reweighting_batch(box_feature[:, 3:6], self.unit_cube.size(0))
        gt_box_pc = provider.transform_pc_batch(self.unit_cube, gt_box_feature)
        with torch.no_grad():
            gt_reweight = provider.get_surface_reweighting_batch(gt_box_feature[:, 3:6], self.unit_cube.size(0))
        dist1, dist2 = self.chamferLoss(gt_box_pc, pred_box_pc)
        loss1 = (dist1 * gt_reweight).sum(dim=1) / (gt_reweight.sum(dim=1) + 1e-12)
        loss2 = (dist2 * pred_reweight).sum(dim=1) / (pred_reweight.sum(dim=1) + 1e-12)
        loss = (loss1 + loss2) / 2
        return loss

    def anchorLossEstimator(self, box_feature, gt_box_feature):
        pred_anchor_pc = provider.transform_pc_batch(self.anchor, box_feature, anchor=True)
        gt_anchor_pc = provider.transform_pc_batch(self.anchor, gt_box_feature, anchor=True)
        dist1, dist2 = self.chamferLoss(gt_anchor_pc, pred_anchor_pc)
        loss = (dist1.mean(dim=1) + dist2.mean(dim=1)) / 2
        return loss

    def isLeafLossEstimator(self, is_leaf_logit, gt_is_leaf):
        return self.bceLoss(is_leaf_logit, gt_is_leaf).view(-1)

    def vectorAdder(self, v1, v2):
        return v1.add_(v2)

    def dummy_box(self, feature):
        return feature.new_tensor([
            0, 0, 0,
            0, 0, 0,
            1, 0, 0,
            0, 1, 0]).unsqueeze(dim=0)

    def decode_structure(self, z, max_depth, fold=None):
        """
        Decode a root code into a tree structure
        """

        if fold is None:
            root_latent = self.sampleDecoder(z)
        else:
            root_latent = fold.add('sampleDecoder', z)

        root = self.decode_node(node_latent=root_latent, max_depth=max_depth, full_label='chair', fold=fold)

        obj = Tree(root=root)

        return obj

    def decode_node(self, node_latent, max_depth, full_label, fold=None, is_leaf=False):

        if node_latent.shape[0] != 1 or fold is not None:
            raise ValueError('Node decoding can currently not be done batched.')

        # this does currently not work with folds, since the node_latent
        # value has to be known here already, so it has to be done un-batched
        # (one sample at a time) for now
        # it also does not use gradients, since the ifs are not differentiable
        # with torch.no_grad():
        is_leaf_logit = self.leafClassifier(node_latent)
        node_is_leaf = is_leaf_logit.item() > 0

        # use maximum depth to avoid potential infinite recursion
        if max_depth <= 1 and not node_is_leaf:
            # print('WARNING: max. tree depth reached, converting node to minimal box node')
            box = self.dummy_box(node_latent)

            return Tree.Node(is_leaf=True, box=box)

        if node_is_leaf or is_leaf:
            box = self.boxDecoder(node_latent)
            ret = Tree.Node(is_leaf=True, full_label=full_label, label=full_label.split('/')[-1])
            ret.set_from_box_quat(box.view(-1))
            return ret
        else:
            (child_feats, child_sem_logits, child_exists_logit,
             edge_exists_logits, edge_feats) = self.childDecoder(node_latent)

            child_sem_logits = child_sem_logits.cpu().numpy().squeeze()

            # decode box
            box = self.boxDecoder(node_latent)

            # children
            child_nodes = []
            child_idx = {}
            for ci in range(child_feats.shape[1]):
                if torch.sigmoid(child_exists_logit[:, ci, :]).item() > 0.5:
                    idx = np.argmax(child_sem_logits[ci, part_name2cids[full_label]])
                    idx = part_name2cids[full_label][idx]
                    child_full_label = part_id2name[idx]
                    child_nodes.append(self.decode_node(
                        node_latent=child_feats[:, ci, :], max_depth=max_depth-1,
                        full_label=child_full_label, fold=fold, is_leaf=(child_full_label not in part_non_leaf_sem_names)))
                    child_idx[ci] = len(child_nodes)-1

            # edges
            child_edges = []
            nz_inds = torch.nonzero(torch.sigmoid(edge_exists_logits) > 0.5)    # NumOfExistingEdges x 4 (batch, from, to, type)
            edge_from = nz_inds[:, 1]
            edge_to = nz_inds[:, 2]
            edge_type = nz_inds[:, 3]

            for i in range(edge_from.numel()):
                cur_edge_from = edge_from[i].item()
                cur_edge_to = edge_to[i].item()
                cur_edge_type = edge_type[i].item()

                if cur_edge_from in child_idx and cur_edge_to in child_idx:
                    if edge_feats is None:
                        edge_params = child_feats.new_zeros(self.conf.edge_feature_size)
                    else:
                        edge_params = edge_feats[0, cur_edge_from, cur_edge_to, cur_edge_type, :]

                    child_edges.append({
                        'part_a': child_idx[cur_edge_from],
                        'part_b': child_idx[cur_edge_to],
                        'type': self.conf.edge_types[cur_edge_type],
                        'params': edge_params})

            node = Tree.Node(is_leaf=False, children=child_nodes, edges=child_edges, full_label=full_label, label=full_label.split('/')[-1])
            node.set_from_box_quat(box.view(-1))
            return node

    def structure_recon_loss(self, z, gt_tree, fold=None, part_id=None, new_box_quat_params=None, interp_sem=None):
        root_latent = self.sampleDecoder(z)
        losses, all_boxes, all_leaf_boxes = self.node_recon_loss(node_latent=root_latent, gt_node=gt_tree.root, fold=fold, \
                part_id=part_id, new_box_quat_params=new_box_quat_params, interp_sem=interp_sem)
        return losses

    # use gt structure, only compute per-node loss (node type and node parameters)
    def node_recon_loss(self, node_latent, gt_node, fold=None, part_id=None, new_box_quat_params=None, interp_sem=None):

        if part_id == gt_node.part_id:
            box = self.boxDecoder(node_latent)
            box_loss = self.boxLossEstimator(box, new_box_quat_params.view(1, -1))
            return {'box': box_loss, 'leaf': torch.zeros_like(box_loss), 'anchor': torch.zeros_like(box_loss), 
                    'exists': torch.zeros_like(box_loss), 'semantic': torch.zeros_like(box_loss),
                    'edge_exists': torch.zeros_like(box_loss), 'edge_feats': torch.zeros_like(box_loss),
                    'sym': torch.zeros_like(box_loss), 'adj': torch.zeros_like(box_loss)}, box, box

        if gt_node.is_leaf:
            box = self.boxDecoder(node_latent)
            box_loss = self.boxLossEstimator(box, gt_node.get_box_quat().view(1, -1))
            anchor_loss = self.anchorLossEstimator(box, gt_node.get_box_quat().view(1, -1))
            is_leaf_logit = self.leafClassifier(node_latent)
            is_leaf_loss = self.isLeafLossEstimator(is_leaf_logit, is_leaf_logit.new_tensor(gt_node.is_leaf).view(1, -1))
            if part_id is not None:
                return {'box': torch.zeros_like(box_loss), 'leaf': torch.zeros_like(box_loss), 'anchor': torch.zeros_like(box_loss), 
                        'exists': torch.zeros_like(box_loss), 'semantic': torch.zeros_like(box_loss),
                        'edge_exists': torch.zeros_like(box_loss), 'edge_feats': torch.zeros_like(box_loss),
                        'sym': torch.zeros_like(box_loss), 'adj': torch.zeros_like(box_loss)}, box, box
            else:
                return {'box': box_loss, 'leaf': is_leaf_loss, 'anchor': anchor_loss, 
                        'exists': torch.zeros_like(box_loss), 'semantic': torch.zeros_like(box_loss),
                        'edge_exists': torch.zeros_like(box_loss), 'edge_feats': torch.zeros_like(box_loss),
                        'sym': torch.zeros_like(box_loss), 'adj': torch.zeros_like(box_loss)}, box, box
        else:
            (child_feats, child_sem_logits, child_exists_logits,
             edge_exists_logits, edge_feats) = self.childDecoder(node_latent)

            feature_len = node_latent.size(1)
            child_pred_boxes = self.boxDecoder(child_feats.view(-1, feature_len))
            num_child_parts = child_pred_boxes.size(0)

            with torch.no_grad():
                child_gt_boxes = torch.cat([child_node.get_box_quat().view(1, -1) for child_node in gt_node.children], dim=0)
                num_gt = child_gt_boxes.size(0)

                child_pred_boxes_tiled = child_pred_boxes.unsqueeze(dim=0).repeat(num_gt, 1, 1)
                child_gt_boxes_tiled = child_gt_boxes.unsqueeze(dim=1).repeat(1, num_child_parts, 1)

                dist_mat = self.boxLossEstimator(child_gt_boxes_tiled.view(-1, 10), child_pred_boxes_tiled.view(-1, 10)).view(-1, num_gt, num_child_parts)

                _, matched_gt_idx, matched_pred_idx = linear_assignment(dist_mat)

                # get edge ground truth
                edge_type_list_gt, edge_feats_list_gt, edge_indices_gt = gt_node.edge_tensors(
                    feat_type=self.conf.edge_feat_type, feat_size=self.conf.edge_feature_size,
                    edge_types=self.conf.edge_types, device=child_feats.device, type_onehot=False)

                gt2pred = {gt_idx: pred_idx for gt_idx, pred_idx in zip(matched_gt_idx, matched_pred_idx)}
                edge_feats_gt = torch.zeros_like(edge_feats) if edge_feats_list_gt is not None else None
                edge_exists_gt = torch.zeros_like(edge_exists_logits)

                sym_from = []; sym_to = []; sym_mat = []; sym_type = []; adj_from = []; adj_to = [];
                for i in range(edge_indices_gt.shape[1]//2):
                    if edge_indices_gt[0, i, 0].item() not in gt2pred or edge_indices_gt[0, i, 1].item() not in gt2pred:
                        # one of the adjacent nodes of the current gt edge was not matched to any node in the prediction, ignore this edge
                        continue
                    edge_from_idx = gt2pred[edge_indices_gt[0, i, 0].item()]
                    edge_to_idx = gt2pred[edge_indices_gt[0, i, 1].item()]
                    edge_exists_gt[:, edge_from_idx, edge_to_idx, edge_type_list_gt[0:1, i]] = 1
                    edge_exists_gt[:, edge_to_idx, edge_from_idx, edge_type_list_gt[0:1, i]] = 1

                    # binary edges
                    if edge_type_list_gt[0, i].item() == 0: # ADJ
                        adj_from.append(edge_from_idx)
                        adj_to.append(edge_to_idx)
                    else:   # REF/TRANS SYM
                        # compute sym-params for the two box preds
                        if edge_type_list_gt[0, i].item() == 1: # ROT_SYM
                            mat1to2, mat2to1 = compute_sym.compute_rot_sym(child_pred_boxes[edge_from_idx].cpu().detach().numpy(), child_pred_boxes[edge_to_idx].cpu().detach().numpy())
                        elif edge_type_list_gt[0, i].item() == 2: # TRANS_SYM
                            mat1to2, mat2to1 = compute_sym.compute_trans_sym(child_pred_boxes[edge_from_idx].cpu().detach().numpy(), child_pred_boxes[edge_to_idx].cpu().detach().numpy())
                        else:   # REF_SYM
                            mat1to2, mat2to1 = compute_sym.compute_ref_sym(child_pred_boxes[edge_from_idx].cpu().detach().numpy(), child_pred_boxes[edge_to_idx].cpu().detach().numpy())
                        sym_from.append(edge_from_idx)
                        sym_to.append(edge_to_idx)
                        sym_mat.append(torch.tensor(mat1to2, dtype=torch.float32, device=self.conf.device).view(1, 3, 4))
                        sym_type.append(edge_type_list_gt[0, i].item())

            # train the current node to be non-leaf
            is_leaf_logit = self.leafClassifier(node_latent)
            is_leaf_loss = self.isLeafLossEstimator(is_leaf_logit, is_leaf_logit.new_tensor(gt_node.is_leaf).view(1, -1))

            # train the current node box to gt
            all_boxes = []; all_leaf_boxes = [];
            box = self.boxDecoder(node_latent)
            all_boxes.append(box)
            if part_id is None:
                box_loss = self.boxLossEstimator(box, gt_node.get_box_quat().view(1, -1))
            else:
                box_loss = 0.0
            anchor_loss = self.anchorLossEstimator(box, gt_node.get_box_quat().view(1, -1))

            # gather information
            child_sem_gt_labels = []
            child_sem_pred_logits = []
            child_box_gt = []
            child_box_pred = []
            child_exists_gt = torch.zeros_like(child_exists_logits)
            for i in range(len(matched_gt_idx)):
                child_sem_gt_labels.append(gt_node.children[matched_gt_idx[i]].get_semantic_id())
                child_sem_pred_logits.append(child_sem_logits[0, matched_pred_idx[i], :].view(1, -1))
                child_box_gt.append(gt_node.children[matched_gt_idx[i]].get_box_quat().view(1, -1))
                child_box_pred.append(child_pred_boxes[matched_pred_idx[i], :].view(1, -1))
                child_exists_gt[:, matched_pred_idx[i], :] = 1

            # train semantic labels
            child_sem_pred_logits = torch.cat(child_sem_pred_logits, dim=0)
            child_sem_gt_labels = torch.tensor(child_sem_gt_labels, dtype=torch.int64, device=child_sem_pred_logits.device)
            semantic_loss = self.semCELoss(child_sem_pred_logits, child_sem_gt_labels) * 0.1
            semantic_loss = semantic_loss.sum()

            # train unused boxes to zeros
            unmatched_boxes = []
            for i in range(num_child_parts):
                if i not in matched_pred_idx:
                    unmatched_boxes.append(child_pred_boxes[i, 3:6].view(1, -1))
            if len(unmatched_boxes) > 0:
                unmatched_boxes = torch.cat(unmatched_boxes, dim=0)
                unused_box_loss = unmatched_boxes.pow(2).sum() * 0.01
            else:
                unused_box_loss = 0.0

            # train exist scores
            child_exists_loss = F.binary_cross_entropy_with_logits(
                input=child_exists_logits, target=child_exists_gt, reduction='none')
            # sum over nodes to be comparable to other losses and to not depend on the maximum number of nodes
            child_exists_loss = child_exists_loss.sum()

            # train edge exists scores
            edge_exists_loss = F.binary_cross_entropy_with_logits(input=edge_exists_logits, target=edge_exists_gt, reduction='none')
            edge_exists_loss = edge_exists_loss.sum()
            edge_exists_loss = edge_exists_loss / (edge_exists_gt.shape[2]*edge_exists_gt.shape[3]) # to make it comparable to the other losses (in particular the child exists loss), which are in the order of the number of child nodes

            # regress edge features
            if edge_feats_gt is not None:
                edge_feats_loss = F.mse_loss(input=edge_feats, target=edge_feats_gt, reduction='none')
                # 0.1 weight and normalized to make it comparable for non-existent edges
                edge_feats_loss[:, 1-edge_exists_gt.byte().view(-1), :] *= 0.1 / (edge_feats_loss.shape[2]*edge_feats_loss.shape[3])
                edge_feats_loss = edge_feats_loss.sum()
            else:
                edge_feats_loss = edge_exists_loss.new_zeros(1)

            # train binary losses
            sym_loss = 0
            if len(sym_from) > 0:
                sym_from_th = torch.tensor(sym_from, dtype=torch.long, device=self.conf.device)
                obb_from = child_pred_boxes[sym_from_th, :]
                with torch.no_grad():
                    reweight_from = provider.get_surface_reweighting_batch(obb_from[:, 3:6], self.unit_cube.size(0))
                pc_from = provider.transform_pc_batch(self.unit_cube, obb_from)
                sym_to_th = torch.tensor(sym_to, dtype=torch.long, device=self.conf.device)
                obb_to = child_pred_boxes[sym_to_th, :]
                with torch.no_grad():
                    reweight_to = provider.get_surface_reweighting_batch(obb_to[:, 3:6], self.unit_cube.size(0))
                pc_to = provider.transform_pc_batch(self.unit_cube, obb_to)
                sym_mat_th = torch.cat(sym_mat, dim=0)
                transformed_pc_from = pc_from.matmul(torch.transpose(sym_mat_th[:, :, :3], 1, 2)) + sym_mat_th[:, :, 3].unsqueeze(dim=1).repeat(1, pc_from.size(1), 1)
                dist1, dist2 = self.chamferLoss(transformed_pc_from, pc_to)
                loss1 = (dist1 * reweight_from).sum(dim=1) / (reweight_from.sum(dim=1) + 1e-12)
                loss2 = (dist2 * reweight_to).sum(dim=1) / (reweight_to.sum(dim=1) + 1e-12)
                loss = loss1 + loss2
                #for i in range(len(sym_from)):
                #    all_v = np.vstack([pc_from[i].cpu().numpy(), pc_to[i].cpu().numpy(), transformed_pc_from[i].cpu().numpy()])
                #    all_l = np.hstack([np.ones((150)), np.ones((150))*2, np.ones((150))*3]).astype(np.int32)
                #    export_ply_with_label(os.path.join('out2', 'from-%d-to-%d-type-%d-loss-%f.ply'%(sym_from[i].item(), sym_to[i].item(), sym_type[i], loss[i].item())), all_v, all_l)
                sym_loss = loss.sum()

            adj_loss = 0
            if len(adj_from) > 0:
                adj_from_th = torch.tensor(adj_from, dtype=torch.long, device=self.conf.device)
                obb_from = child_pred_boxes[adj_from_th, :]
                pc_from = provider.transform_pc_batch(self.unit_cube, obb_from)
                adj_to_th = torch.tensor(adj_to, dtype=torch.long, device=self.conf.device)
                obb_to = child_pred_boxes[adj_to_th, :]
                pc_to = provider.transform_pc_batch(self.unit_cube, obb_to)
                dist1, dist2 = self.chamferLoss(pc_from, pc_to)
                loss = (dist1.min(dim=1)[0] + dist2.min(dim=1)[0])
                adj_loss = loss.sum()

            # call children + aggregate losses
            pred2allboxes = dict(); pred2allleafboxes = dict();
            for i in range(len(matched_gt_idx)):
                child_losses, child_all_boxes, child_all_leaf_boxes = self.node_recon_loss(
                    node_latent=child_feats[:, matched_pred_idx[i], :],
                    gt_node=gt_node.children[matched_gt_idx[i]], fold=fold, part_id=part_id, new_box_quat_params=new_box_quat_params)
                pred2allboxes[matched_pred_idx[i]] = child_all_boxes
                pred2allleafboxes[matched_pred_idx[i]] = child_all_leaf_boxes
                all_boxes.append(child_all_boxes)
                all_leaf_boxes.append(child_all_leaf_boxes)
                box_loss = box_loss + child_losses['box']
                anchor_loss = anchor_loss + child_losses['anchor'] 
                is_leaf_loss = is_leaf_loss + child_losses['leaf']
                child_exists_loss = child_exists_loss + child_losses['exists']
                semantic_loss = semantic_loss + child_losses['semantic']
                edge_exists_loss = edge_exists_loss + child_losses['edge_exists']
                edge_feats_loss = edge_feats_loss + child_losses['edge_feats']
                sym_loss = sym_loss + child_losses['sym']
                adj_loss = adj_loss + child_losses['adj']

            # train inside children relationship
            for i in range(len(sym_from)):
                s1 = pred2allboxes[sym_from[i]].size(0)
                s2 = pred2allboxes[sym_to[i]].size(0)
                if s1 > 1 and s2 > 1:
                    obbs_from = pred2allboxes[sym_from[i]][1:, :]
                    obbs_to = pred2allboxes[sym_to[i]][1:, :]
                    pc_from = provider.transform_pc_batch(self.unit_cube, obbs_from).view(-1, 3)
                    pc_to = provider.transform_pc_batch(self.unit_cube, obbs_to).view(-1, 3)
                    transformed_pc_from = pc_from.matmul(torch.transpose(sym_mat[i][0, :, :3], 0, 1)) + sym_mat[i][0, :, 3].unsqueeze(dim=0).repeat(pc_from.size(0), 1)
                    dist1, dist2 = self.chamferLoss(transformed_pc_from.view(1, -1, 3), pc_to.view(1, -1, 3))
                    sym_loss += (dist1.mean() + dist2.mean()) * (s1 + s2) / 2

            for i in range(len(adj_from)):
                if pred2allboxes[adj_from[i]].size(0) > pred2allleafboxes[adj_from[i]].size(0) or pred2allboxes[adj_to[i]].size(0) > pred2allleafboxes[adj_to[i]].size(0):
                    obbs_from = pred2allleafboxes[adj_from[i]]
                    obbs_to = pred2allleafboxes[adj_to[i]]
                    pc_from = provider.transform_pc_batch(self.unit_cube, obbs_from).view(1, -1, 3)
                    pc_to = provider.transform_pc_batch(self.unit_cube, obbs_to).view(1, -1, 3)
                    dist1, dist2 = self.chamferLoss(pc_from, pc_to)
                    adj_loss += dist1.min() + dist2.min()

            if part_id is not None:
                return {'box': box_loss, 'leaf': is_leaf_loss, 'anchor': anchor_loss, 
                        'exists': child_exists_loss, 'semantic': semantic_loss,
                        'edge_exists': edge_exists_loss, 'edge_feats': edge_feats_loss,
                        'sym': sym_loss, 'adj': adj_loss}, torch.cat(all_boxes, dim=0), torch.cat(all_leaf_boxes, dim=0)
            else:
                return {'box': box_loss + unused_box_loss, 'leaf': is_leaf_loss, 'anchor': anchor_loss, 
                        'exists': child_exists_loss, 'semantic': semantic_loss,
                        'edge_exists': edge_exists_loss, 'edge_feats': edge_feats_loss,
                        'sym': sym_loss, 'adj': adj_loss}, torch.cat(all_boxes, dim=0), torch.cat(all_leaf_boxes, dim=0)

