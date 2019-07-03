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


class PartFeatSampler(nn.Module):

    def __init__(self, feature_size, probabilistic=True):
        super(PartFeatSampler, self).__init__()

        self.probabilistic = probabilistic

        self.mlp2mu = nn.Linear(feature_size, feature_size)
        self.mlp2var = nn.Linear(feature_size, feature_size)

    def forward(self, x):
        mu = self.mlp2mu(x)

        if self.probabilistic:
            logvar = self.mlp2var(x)
            # log(variance) to standard deviation
            std = logvar.mul(0.5).exp_()
            # standard normal distributed noise
            eps = torch.randn_like(std)

            # KL-divergence
            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)

            return torch.cat([eps.mul(std).add_(mu), kld], 1)
        else:
            return mu


class PartEncoder(nn.Module):

    def __init__(self, feat_len):
        super(PartEncoder, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, feat_len, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(feat_len)

        self.sampler = PartFeatSampler(feature_size=feat_len, probabilistic=False)

    def forward(self, pc):

        net = pc.transpose(2, 1)
        net = torch.relu(self.bn1(self.conv1(net)))
        net = torch.relu(self.bn2(self.conv2(net)))
        net = torch.relu(self.bn3(self.conv3(net)))
        net = torch.relu(self.bn4(self.conv4(net)))

        net = net.max(dim=2)[0]
        net = self.sampler(net)

        return net


class NodeEncoder(nn.Module):

    def __init__(self, geo_feat_len, node_feat_len):
        super(NodeEncoder, self).__init__()

        self.part_encoder = PartEncoder(geo_feat_len)

        self.mlp1 = nn.Linear(4, geo_feat_len)
        self.mlp2 = nn.Linear(2 * geo_feat_len, node_feat_len)

    def forward(self, geo):

        num_point = geo.size(1)

        center = geo.mean(dim=1)
        geo = geo - center.unsqueeze(dim=1).repeat(1, num_point, 1)
        scale = geo.pow(2).sum(dim=2).max(dim=1)[0].sqrt().view(-1, 1)
        box_feat = torch.cat([center, scale], dim=1)
        box_feat = torch.relu(self.mlp1(box_feat))

        geo = geo / scale.unsqueeze(dim=1).repeat(1, num_point, 3)
        geo_feat = self.part_encoder(geo)

        all_feat = torch.cat([box_feat, geo_feat], dim=1)
        all_feat = torch.relu(self.mlp2(all_feat))

        return all_feat, geo_feat


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

        self.node_encoder = NodeEncoder(geo_feat_len=config.geo_feat_size, node_feat_len=config.feature_size)
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

    def nodeEncoder(self, pc):
        return self.node_encoder(pc)

    def childEncoder(self, child_feats, child_exists, edge_type_onehot=None, edge_feats=None, edge_indices=None):
        if 'gnn' in self.child_encoder_type:
            return self.child_encoder(child_feats, child_exists, edge_type_onehot, edge_feats, edge_indices)
        else:
            return self.child_encoder(child_feats, child_exists)

    def sampleEncoder(self, feature):
        return self.sample_encoder(feature)

    def encode_node(self, node, fold=None):

        all_feat, geo_feat = self.nodeEncoder(node.geo)
        node.geo_feat = geo_feat

        if node.is_leaf:
            return all_feat
        else:
            # get features of all children
            child_feats = []
            for child in node.children:
                cur_child_feat = torch.cat([self.encode_node(node=child, fold=fold), child.get_semantic_one_hot()], dim=1)
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

    def encode_structure(self, obj, fold=None):
        root_latent = self.encode_node(node=obj.root, fold=fold)
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


class PartDecoder(nn.Module):

    def __init__(self, feat_len, num_point):
        super(PartDecoder, self).__init__()

        self.num_point = num_point

        self.mlp1 = nn.Linear(feat_len, feat_len)
        self.mlp2 = nn.Linear(feat_len, feat_len)
        self.mlp3 = nn.Linear(feat_len, num_point*3)

        self.bn1 = nn.BatchNorm1d(feat_len)
        self.bn2 = nn.BatchNorm1d(feat_len)

    def forward(self, net):

        net = torch.relu(self.bn1(self.mlp1(net)))
        net = torch.relu(self.bn2(self.mlp2(net)))
        net = self.mlp3(net).view(-1, self.num_point, 3)

        return net


class NodeDecoder(nn.Module):

    def __init__(self, geo_feat_len, node_feat_len, num_point):
        super(NodeDecoder, self).__init__()

        self.mlp1 = nn.Linear(node_feat_len, 3)
        self.mlp2 = nn.Linear(node_feat_len, 1)
        self.mlp3 = nn.Linear(node_feat_len, geo_feat_len)

        self.part_decoder = PartDecoder(geo_feat_len, num_point)

    def forward(self, net):

        geo_center = torch.tanh(self.mlp1(net))
        geo_scale = torch.sigmoid(self.mlp2(net))

        geo_feat = self.mlp3(net)
        geo_local = self.part_decoder(geo_feat)

        return geo_local, geo_center, geo_scale, geo_feat


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

        self.node_decoder = NodeDecoder(geo_feat_len=config.geo_feat_size, node_feat_len=config.feature_size, num_point=config.num_point)

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
        self.semCELoss = nn.CrossEntropyLoss(reduction='none')
        self.mseLoss = nn.MSELoss(reduction='none')
        self.chamferLoss = ChamferDistance()
        self.child_decoder_type = child_decoder_type

        self.register_buffer('unit_cube', torch.from_numpy(load_pts('cube.pts')))

    def nodeDecoder(self, feature):
        return self.node_decoder(feature)

    def geoToGlobal(self, geo_local, geo_center, geo_scale):
        num_point = geo_local.size(1)
        return geo_local * geo_scale.unsqueeze(dim=1).repeat(1, num_point, 3) + geo_center.unsqueeze(dim=1).repeat(1, num_point, 1)

    def chamferDist(self, pc1, pc2):
        dist1, dist2 = self.chamferLoss(pc1, pc2)
        return (dist1.mean(dim=1) + dist2.mean(dim=1)) / 2

    def childDecoder(self, feature):
        return self.child_decoder(feature)

    def sampleDecoder(self, feature):
        return self.sample_decoder(feature)

    def leafClassifier(self, feature):
        return self.leaf_classifier(feature)

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
            print('WARNING: max. tree depth reached, converting node to minimal box node')
            box = self.dummy_box(node_latent)
            return Tree.Node(is_leaf=True, box=box)

        geo_local, geo_center, geo_scale, _ = self.nodeDecoder(node_latent)
        geo_global = self.geoToGlobal(geo_local, geo_center, geo_scale)

        if node_is_leaf or is_leaf:
            return Tree.Node(is_leaf=True, full_label=full_label, label=full_label.split('/')[-1], geo=geo_global)
        else:
            (child_feats, child_sem_logits, child_exists_logit,
             edge_exists_logits, edge_feats) = self.childDecoder(node_latent)

            child_sem_logits = child_sem_logits.cpu().numpy().squeeze()

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

            return Tree.Node(is_leaf=False, children=child_nodes, edges=child_edges, full_label=full_label, label=full_label.split('/')[-1], geo=geo_global)

    def structure_recon_loss(self, z, gt_tree, fold=None):
        root_latent = self.sampleDecoder(z)
        losses, _, _ = self.node_recon_loss(node_latent=root_latent, gt_node=gt_tree.root, fold=fold)
        return losses

    # use gt structure, only compute per-node loss (node type and node parameters)
    def node_recon_loss(self, node_latent, gt_node, fold=None):

        gt_geo = gt_node.geo
        gt_geo_feat = gt_node.geo_feat
        gt_center = gt_geo.mean(dim=1)
        gt_scale = (gt_geo - gt_center.unsqueeze(dim=1).repeat(1, self.conf.num_point, 1)).pow(2).sum(dim=2).max(dim=1)[0].sqrt().view(1, 1)

        geo_local, geo_center, geo_scale, geo_feat = self.nodeDecoder(node_latent)
        geo_global = self.geoToGlobal(geo_local, geo_center, geo_scale)

        latent_loss = self.mseLoss(geo_feat, gt_geo_feat).mean()
        center_loss = self.mseLoss(geo_center, gt_center).mean()
        scale_loss = self.mseLoss(geo_scale, gt_scale).mean()
        geo_loss = self.chamferDist(geo_global, gt_geo)

        if gt_node.is_leaf:
            is_leaf_logit = self.leafClassifier(node_latent)
            is_leaf_loss = self.isLeafLossEstimator(is_leaf_logit, is_leaf_logit.new_tensor(gt_node.is_leaf).view(1, -1))
            return {'leaf': is_leaf_loss, 'geo': geo_loss, 'center': center_loss, 'scale': scale_loss, 'latent': latent_loss,
                    'exists': torch.zeros_like(geo_loss), 'semantic': torch.zeros_like(geo_loss),
                    'edge_exists': torch.zeros_like(geo_loss), 'edge_feats': torch.zeros_like(geo_loss),
                    'sym': torch.zeros_like(geo_loss), 'adj': torch.zeros_like(geo_loss)}, geo_global, geo_global
        else:
            (child_feats, child_sem_logits, child_exists_logits,
             edge_exists_logits, edge_feats) = self.childDecoder(node_latent)

            feature_len = node_latent.size(1)

            all_geo = []; all_leaf_geo = [];
            all_geo.append(geo_global)

            child_pred_geo_local, child_pred_geo_center, child_pred_geo_scale, _ = self.nodeDecoder(child_feats.view(-1, feature_len))
            child_pred_geo = self.geoToGlobal(child_pred_geo_local, child_pred_geo_center, child_pred_geo_scale)
            num_pred = child_pred_geo.size(0)

            with torch.no_grad():
                child_gt_geo = torch.cat([child_node.geo for child_node in gt_node.children], dim=0)
                num_gt = child_gt_geo.size(0)

                child_pred_geo_tiled = child_pred_geo.unsqueeze(dim=0).repeat(num_gt, 1, 1, 1)
                child_gt_geo_tiled = child_gt_geo.unsqueeze(dim=1).repeat(1, num_pred, 1, 1)

                dist_mat = self.chamferDist(child_pred_geo_tiled.view(-1, self.conf.num_point, 3), child_gt_geo_tiled.view(-1, self.conf.num_point, 3)).view(1, num_gt, num_pred)
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
                        obb1 = compute_sym.compute_obb(child_pred_geo[edge_from_idx].cpu().detach().numpy())
                        obb2 = compute_sym.compute_obb(child_pred_geo[edge_to_idx].cpu().detach().numpy())
                        if edge_type_list_gt[0, i].item() == 1: # ROT_SYM
                            mat1to2, mat2to1 = compute_sym.compute_rot_sym(obb1, obb2)
                        elif edge_type_list_gt[0, i].item() == 2: # TRANS_SYM
                            mat1to2, mat2to1 = compute_sym.compute_trans_sym(obb1, obb2)
                        else:   # REF_SYM
                            mat1to2, mat2to1 = compute_sym.compute_ref_sym(obb1, obb2)
                        sym_from.append(edge_from_idx)
                        sym_to.append(edge_to_idx)
                        sym_mat.append(torch.tensor(mat1to2, dtype=torch.float32, device=self.conf.device).view(1, 3, 4))
                        sym_type.append(edge_type_list_gt[0, i].item())

            # train the current node to be non-leaf
            is_leaf_logit = self.leafClassifier(node_latent)
            is_leaf_loss = self.isLeafLossEstimator(is_leaf_logit, is_leaf_logit.new_tensor(gt_node.is_leaf).view(1, -1))

            # gather information
            child_sem_gt_labels = []
            child_sem_pred_logits = []
            child_exists_gt = torch.zeros_like(child_exists_logits)
            for i in range(len(matched_gt_idx)):
                child_sem_gt_labels.append(gt_node.children[matched_gt_idx[i]].get_semantic_id())
                child_sem_pred_logits.append(child_sem_logits[0, matched_pred_idx[i], :].view(1, -1))
                child_exists_gt[:, matched_pred_idx[i], :] = 1
            
            # train unused geo to zero
            unmatched_scales = []
            for i in range(num_pred):
                if i not in matched_pred_idx:
                    unmatched_scales.append(child_pred_geo_scale[i:i+1])
            if len(unmatched_scales) > 0:
                unmatched_scales = torch.cat(unmatched_scales, dim=0)
                unused_geo_loss = unmatched_scales.pow(2).sum()
            else:
                unused_geo_loss = 0.0

            # train semantic labels
            child_sem_pred_logits = torch.cat(child_sem_pred_logits, dim=0)
            child_sem_gt_labels = torch.tensor(child_sem_gt_labels, dtype=torch.int64, device=child_sem_pred_logits.device)
            semantic_loss = self.semCELoss(child_sem_pred_logits, child_sem_gt_labels)
            semantic_loss = semantic_loss.sum()

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
                pc_from = child_pred_geo[sym_from_th, :]
                sym_to_th = torch.tensor(sym_to, dtype=torch.long, device=self.conf.device)
                pc_to = child_pred_geo[sym_to_th, :]
                sym_mat_th = torch.cat(sym_mat, dim=0)
                transformed_pc_from = pc_from.matmul(torch.transpose(sym_mat_th[:, :, :3], 1, 2)) + sym_mat_th[:, :, 3].unsqueeze(dim=1).repeat(1, pc_from.size(1), 1)
                loss = self.chamferDist(transformed_pc_from, pc_to) * 2
                sym_loss = loss.sum()

            adj_loss = 0
            if len(adj_from) > 0:
                adj_from_th = torch.tensor(adj_from, dtype=torch.long, device=self.conf.device)
                pc_from = child_pred_geo[adj_from_th, :]
                adj_to_th = torch.tensor(adj_to, dtype=torch.long, device=self.conf.device)
                pc_to = child_pred_geo[adj_to_th, :]
                dist1, dist2 = self.chamferLoss(pc_from, pc_to)
                loss = (dist1.min(dim=1)[0] + dist2.min(dim=1)[0])
                adj_loss = loss.sum()

            # call children + aggregate losses
            pred2allgeo = dict(); pred2allleafgeo = dict();
            for i in range(len(matched_gt_idx)):
                child_losses, child_all_geo, child_all_leaf_geo = self.node_recon_loss(
                    node_latent=child_feats[:, matched_pred_idx[i], :],
                    gt_node=gt_node.children[matched_gt_idx[i]], fold=fold)
                pred2allgeo[matched_pred_idx[i]] = child_all_geo
                pred2allleafgeo[matched_pred_idx[i]] = child_all_leaf_geo
                all_geo.append(child_all_geo)
                all_leaf_geo.append(child_all_leaf_geo)
                latent_loss = latent_loss + child_losses['latent']
                geo_loss = geo_loss + child_losses['geo']
                center_loss = center_loss + child_losses['center']
                scale_loss = scale_loss + child_losses['scale']
                is_leaf_loss = is_leaf_loss + child_losses['leaf']
                child_exists_loss = child_exists_loss + child_losses['exists']
                semantic_loss = semantic_loss + child_losses['semantic']
                edge_exists_loss = edge_exists_loss + child_losses['edge_exists']
                edge_feats_loss = edge_feats_loss + child_losses['edge_feats']
                sym_loss = sym_loss + child_losses['sym']
                adj_loss = adj_loss + child_losses['adj']

            # train inside children relationship
            for i in range(len(sym_from)):
                s1 = pred2allgeo[sym_from[i]].size(0)
                s2 = pred2allgeo[sym_to[i]].size(0)
                if s1 > 1 and s2 > 1:
                    pc_from = pred2allgeo[sym_from[i]][1:, :, :].view(-1, 3)
                    pc_to = pred2allgeo[sym_to[i]][1:, :, :].view(-1, 3)
                    transformed_pc_from = pc_from.matmul(torch.transpose(sym_mat[i][0, :, :3], 0, 1)) + sym_mat[i][0, :, 3].unsqueeze(dim=0).repeat(pc_from.size(0), 1)
                    dist1, dist2 = self.chamferLoss(transformed_pc_from.view(1, -1, 3), pc_to.view(1, -1, 3))
                    sym_loss += (dist1.mean() + dist2.mean()) * (s1 + s2) / 2

            for i in range(len(adj_from)):
                if pred2allgeo[adj_from[i]].size(0) > pred2allleafgeo[adj_from[i]].size(0) or pred2allgeo[adj_to[i]].size(0) > pred2allleafgeo[adj_to[i]].size(0):
                    pc_from = pred2allleafgeo[adj_from[i]].view(1, -1, 3)
                    pc_to = pred2allleafgeo[adj_to[i]].view(1, -1, 3)
                    dist1, dist2 = self.chamferLoss(pc_from, pc_to)
                    adj_loss += dist1.min() + dist2.min()

            return {'leaf': is_leaf_loss, 'geo': geo_loss + unused_geo_loss, 'center': center_loss, 'scale': scale_loss, 'latent': latent_loss, 
                    'exists': child_exists_loss, 'semantic': semantic_loss,
                    'edge_exists': edge_exists_loss, 'edge_feats': edge_feats_loss,
                    'sym': sym_loss, 'adj': adj_loss}, torch.cat(all_geo, dim=0), torch.cat(all_leaf_geo, dim=0)

