"""
    This file defines the box-represented shape VAE/AE model.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch_scatter
import compute_sym
from chamfer_distance import ChamferDistance
from data import Tree
from utils import linear_assignment, load_pts, transform_pc_batch, get_surface_reweighting_batch


class Sampler(nn.Module):

    def __init__(self, feature_size, hidden_size, probabilistic=True):
        super(Sampler, self).__init__()
        self.probabilistic = probabilistic

        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, feature_size)
        self.mlp2var = nn.Linear(hidden_size, feature_size)

    def forward(self, x):
        encode = torch.relu(self.mlp1(x))
        mu = self.mlp2mu(encode)

        if self.probabilistic:
            logvar = self.mlp2var(encode)
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)

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


class GNNChildEncoder(nn.Module):

    def __init__(self, node_feat_size, hidden_size, node_symmetric_type, \
            edge_symmetric_type, num_iterations, edge_type_num):
        super(GNNChildEncoder, self).__init__()

        self.node_symmetric_type = node_symmetric_type
        self.edge_symmetric_type = edge_symmetric_type
        self.num_iterations = num_iterations
        self.edge_type_num = edge_type_num

        self.child_op = nn.Linear(node_feat_size + Tree.num_sem, hidden_size)
        self.node_edge_op = torch.nn.ModuleList()
        for i in range(self.num_iterations):
            self.node_edge_op.append(nn.Linear(hidden_size*2+edge_type_num, hidden_size))

        self.parent_op = nn.Linear(hidden_size*(self.num_iterations+1), node_feat_size)

    """
        Input Arguments:
            child feats: b x max_childs x feat_dim
            child exists: b x max_childs x 1
            edge_type_onehot: b x num_edges x edge_type_num
            edge_indices: b x num_edges x 2
    """
    def forward(self, child_feats, child_exists, edge_type_onehot, edge_indices):
        batch_size = child_feats.shape[0]
        max_childs = child_feats.shape[1]
        num_edges = edge_indices.shape[1]

        if batch_size != 1:
            raise ValueError('Currently only a single batch is supported.')

        # perform MLP for child features
        child_feats = torch.relu(self.child_op(child_feats))
        hidden_size = child_feats.size(-1)

        # zero out non-existent children
        child_feats = child_feats * child_exists
        child_feats = child_feats.view(1, max_childs, -1)

        # combine node features before and after message-passing into one parent feature
        iter_parent_feats = []
        if self.node_symmetric_type == 'max':
            iter_parent_feats.append(child_feats.max(dim=1)[0])
        elif self.node_symmetric_type == 'sum':
            iter_parent_feats.append(child_feats.sum(dim=1))
        elif self.node_symmetric_type == 'avg':
            iter_parent_feats.append(child_feats.sum(dim=1) / child_exists.sum(dim=1))
        else:
            raise ValueError(f'Unknown node symmetric type: {self.node_symmetric_type}')

        if self.num_iterations > 0 and num_edges > 0:
            edge_feats = edge_type_onehot

        edge_indices_from = edge_indices[:, :, 0].view(-1, 1).expand(-1, hidden_size)

        # perform Graph Neural Network for message-passing among sibling nodes
        for i in range(self.num_iterations):
            if num_edges > 0:
                # MLP for edge features concatenated with adjacent node features
                node_edge_feats = torch.cat([
                    child_feats[0:1, edge_indices[0, :, 0], :], # start node features
                    child_feats[0:1, edge_indices[0, :, 1], :], # end node features
                    edge_feats], dim=2) # edge features

                node_edge_feats = node_edge_feats.view(num_edges, -1)
                node_edge_feats = torch.relu(self.node_edge_op[i](node_edge_feats))
                node_edge_feats = node_edge_feats.view(num_edges, -1)

                # aggregate information from neighboring nodes
                new_child_feats = child_feats.new_zeros(max_childs, hidden_size)
                if self.edge_symmetric_type == 'max':
                    new_child_feats, _ = torch_scatter.scatter_max(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                elif self.edge_symmetric_type == 'sum':
                    new_child_feats = torch_scatter.scatter_add(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                elif self.edge_symmetric_type == 'avg':
                    new_child_feats = torch_scatter.scatter_mean(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                else:
                    raise ValueError(f'Unknown edge symmetric type: {self.edge_symmetric_type}')

                child_feats = new_child_feats.view(1, max_childs, hidden_size)

            # combine node features before and after message-passing into one parent feature
            if self.node_symmetric_type == 'max':
                iter_parent_feats.append(child_feats.max(dim=1)[0])
            elif self.node_symmetric_type == 'sum':
                iter_parent_feats.append(child_feats.sum(dim=1))
            elif self.node_symmetric_type == 'avg':
                iter_parent_feats.append(child_feats.sum(dim=1) / child_exists.sum(dim=1))
            else:
                raise ValueError(f'Unknown node symmetric type: {self.node_symmetric_type}')

        # concatenation of the parent features from all iterations (as in GIN, like skip connections)
        parent_feat = torch.cat(iter_parent_feats, dim=1)

        # back to standard feature space size
        parent_feat = torch.relu(self.parent_op(parent_feat))

        return parent_feat


class RecursiveEncoder(nn.Module):

    def __init__(self, config, variational=False, probabilistic=True):
        super(RecursiveEncoder, self).__init__()
        self.conf = config

        self.box_encoder = BoxEncoder(feature_size=config.feature_size)

        self.child_encoder = GNNChildEncoder(
            node_feat_size=config.feature_size,
            hidden_size=config.hidden_size,
            node_symmetric_type=config.node_symmetric_type,
            edge_symmetric_type=config.edge_symmetric_type,
            num_iterations=config.num_gnn_iterations,
            edge_type_num=len(config.edge_types))

        if variational:
            self.sample_encoder = Sampler(feature_size=config.feature_size, \
                    hidden_size=config.hidden_size, probabilistic=probabilistic)

    def encode_node(self, node):
        if node.is_leaf:
            return self.box_encoder(node.get_box_quat().view(1, -1))
        else:
            # get features of all children
            child_feats = []
            for child in node.children:
                cur_child_feat = torch.cat([self.encode_node(child), child.get_semantic_one_hot()], dim=1)
                child_feats.append(cur_child_feat.unsqueeze(dim=1))
            child_feats = torch.cat(child_feats, dim=1)

            if child_feats.shape[1] > self.conf.max_child_num:
                raise ValueError('Node has too many children.')

            # pad with zeros
            if child_feats.shape[1] < self.conf.max_child_num:
                padding = child_feats.new_zeros(child_feats.shape[0], \
                        self.conf.max_child_num-child_feats.shape[1], child_feats.shape[2])
                child_feats = torch.cat([child_feats, padding], dim=1)

            # 1 if the child exists, 0 if it is padded
            child_exists = child_feats.new_zeros(child_feats.shape[0], self.conf.max_child_num, 1)
            child_exists[:, :len(node.children), :] = 1

            # get feature of current node (parent of the children)
            edge_type_onehot, edge_indices = node.edge_tensors(
                edge_types=self.conf.edge_types, device=child_feats.device, type_onehot=True)

            return self.child_encoder(child_feats, child_exists, edge_type_onehot, edge_indices)

    def encode_structure(self, obj):
        root_latent = self.encode_node(obj.root)
        return self.sample_encoder(root_latent)


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

    def __init__(self, node_feat_size, hidden_size, max_child_num, \
            edge_symmetric_type, num_iterations, edge_type_num):
        super(GNNChildDecoder, self).__init__()

        self.max_child_num = max_child_num
        self.hidden_size = hidden_size
        self.edge_symmetric_type = edge_symmetric_type
        self.num_iterations = num_iterations
        self.edge_type_num = edge_type_num

        self.mlp_parent = nn.Linear(node_feat_size, hidden_size*max_child_num)
        self.mlp_exists = nn.Linear(hidden_size, 1)
        self.mlp_sem = nn.Linear(hidden_size, Tree.num_sem)
        self.mlp_child = nn.Linear(hidden_size, node_feat_size)
        self.mlp_edge_latent = nn.Linear(hidden_size*2, hidden_size)

        self.mlp_edge_exists = nn.ModuleList()
        for i in range(edge_type_num):
            self.mlp_edge_exists.append(nn.Linear(hidden_size, 1))

        self.node_edge_op = torch.nn.ModuleList()
        for i in range(self.num_iterations):
            self.node_edge_op.append(nn.Linear(hidden_size*3+edge_type_num, hidden_size))

        self.mlp_child = nn.Linear(hidden_size*(self.num_iterations+1), hidden_size)
        self.mlp_sem = nn.Linear(hidden_size, Tree.num_sem)
        self.mlp_child2 = nn.Linear(hidden_size, node_feat_size)

    def forward(self, parent_feature):
        batch_size = parent_feature.shape[0]
        feat_size = parent_feature.shape[1]

        if batch_size != 1:
            raise ValueError('Only batch size 1 supported for now.')

        parent_feature = torch.relu(self.mlp_parent(parent_feature))
        child_feats = parent_feature.view(batch_size, self.max_child_num, self.hidden_size)

        # node existence
        child_exists_logits = self.mlp_exists(child_feats.view(batch_size*self.max_child_num, self.hidden_size))
        child_exists_logits = child_exists_logits.view(batch_size, self.max_child_num, 1)

        # edge features
        edge_latents = torch.cat([
            child_feats.view(batch_size, self.max_child_num, 1, feat_size).expand(-1, -1, self.max_child_num, -1),
            child_feats.view(batch_size, 1, self.max_child_num, feat_size).expand(-1, self.max_child_num, -1, -1)
            ], dim=3)
        edge_latents = torch.relu(self.mlp_edge_latent(edge_latents))

        # edge existence prediction
        edge_exists_logits_per_type = []
        for i in range(self.edge_type_num):
            edge_exists_logits_cur_type = self.mlp_edge_exists[i](edge_latents).view(\
                    batch_size, self.max_child_num, self.max_child_num, 1)
            edge_exists_logits_per_type.append(edge_exists_logits_cur_type)
        edge_exists_logits = torch.cat(edge_exists_logits_per_type, dim=3)

        """
            decoding stage message passing
            there are several possible versions, this is a simple one:
            use a fixed set of edges, consisting of existing edges connecting existing nodes
            this set of edges does not change during iterations
            iteratively update the child latent features
            then use these child latent features to compute child features and semantics
        """
        # get edges that exist between nodes that exist
        edge_indices = torch.nonzero(edge_exists_logits > 0)
        edge_types = edge_indices[:, 3]
        edge_indices = edge_indices[:, 1:3]
        nodes_exist_mask = (child_exists_logits[0, edge_indices[:, 0], 0] > 0) \
                & (child_exists_logits[0, edge_indices[:, 1], 0] > 0)
        edge_indices = edge_indices[nodes_exist_mask, :]
        edge_types = edge_types[nodes_exist_mask]

        # get latent features for the edges
        edge_feats_mp = edge_latents[0:1, edge_indices[:, 0], edge_indices[:, 1], :]

        # append edge type to edge features, so the network has information which
        # of the possibly multiple edges between two nodes it is working with
        edge_type_logit = edge_exists_logits[0:1, edge_indices[:, 0], edge_indices[:, 1], :]
        edge_type_logit = edge_feats_mp.new_zeros(edge_feats_mp.shape[:2]+(self.edge_type_num,))
        edge_type_logit[0:1, range(edge_type_logit.shape[1]), edge_types] = \
                edge_exists_logits[0:1, edge_indices[:, 0], edge_indices[:, 1], edge_types]
        edge_feats_mp = torch.cat([edge_feats_mp, edge_type_logit], dim=2)

        num_edges = edge_indices.shape[0]
        max_childs = child_feats.shape[1]

        iter_child_feats = [child_feats] # zeroth iteration

        if self.num_iterations > 0 and num_edges > 0:
            edge_indices_from = edge_indices[:, 0].view(-1, 1).expand(-1, self.hidden_size)

        for i in range(self.num_iterations):
            if num_edges > 0:
                node_edge_feats = torch.cat([
                    child_feats[0:1, edge_indices[:, 0], :], # start node features
                    child_feats[0:1, edge_indices[:, 1], :], # end node features
                    edge_feats_mp], dim=2) # edge features

                node_edge_feats = node_edge_feats.view(num_edges, -1)
                node_edge_feats = torch.relu(self.node_edge_op[i](node_edge_feats))

                # aggregate information from neighboring nodes
                new_child_feats = child_feats.new_zeros(max_childs, self.hidden_size)
                if self.edge_symmetric_type == 'max':
                    new_child_feats, _ = torch_scatter.scatter_max(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                elif self.edge_symmetric_type == 'sum':
                    new_child_feats = torch_scatter.scatter_add(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                elif self.edge_symmetric_type == 'avg':
                    new_child_feats = torch_scatter.scatter_mean(node_edge_feats, edge_indices_from, dim=0, out=new_child_feats)
                else:
                    raise ValueError(f'Unknown edge symmetric type: {self.edge_symmetric_type}')

                child_feats = new_child_feats.view(1, max_childs, self.hidden_size)

            # save child features of this iteration
            iter_child_feats.append(child_feats)

        # concatenation of the child features from all iterations (as in GIN, like skip connections)
        child_feats = torch.cat(iter_child_feats, dim=2)

        # transform concatenation back to original feature space size
        child_feats = child_feats.view(-1, self.hidden_size*(self.num_iterations+1))
        child_feats = torch.relu(self.mlp_child(child_feats))
        child_feats = child_feats.view(batch_size, self.max_child_num, self.hidden_size)

        # node semantics
        child_sem_logits = self.mlp_sem(child_feats.view(-1, self.hidden_size))
        child_sem_logits = child_sem_logits.view(batch_size, self.max_child_num, Tree.num_sem)

        # node features
        child_feats = self.mlp_child2(child_feats.view(-1, self.hidden_size))
        child_feats = child_feats.view(batch_size, self.max_child_num, feat_size)
        child_feats = torch.relu(child_feats)

        return child_feats, child_sem_logits, child_exists_logits, edge_exists_logits


class RecursiveDecoder(nn.Module):

    def __init__(self, config):
        super(RecursiveDecoder, self).__init__()

        self.conf = config

        self.box_decoder = BoxDecoder(config.feature_size, config.hidden_size)

        self.child_decoder = GNNChildDecoder(
            node_feat_size=config.feature_size,
            hidden_size=config.hidden_size,
            max_child_num=config.max_child_num,
            edge_symmetric_type=config.edge_symmetric_type,
            num_iterations=config.num_dec_gnn_iterations,
            edge_type_num=len(config.edge_types))

        self.sample_decoder = SampleDecoder(config.feature_size, config.hidden_size)

        self.leaf_classifier = LeafClassifier(config.feature_size, config.hidden_size)

        self.bceLoss = nn.BCEWithLogitsLoss(reduction='none')
        self.chamferLoss = ChamferDistance()
        self.semCELoss = nn.CrossEntropyLoss(reduction='none')

        self.register_buffer('unit_cube', torch.from_numpy(load_pts('cube.pts')))
        self.register_buffer('anchor', torch.from_numpy(load_pts('anchor.pts')))

    def boxLossEstimator(self, box_feature, gt_box_feature):
        pred_box_pc = transform_pc_batch(self.unit_cube, box_feature)
        with torch.no_grad():
            pred_reweight = get_surface_reweighting_batch(box_feature[:, 3:6], self.unit_cube.size(0))
        gt_box_pc = transform_pc_batch(self.unit_cube, gt_box_feature)
        with torch.no_grad():
            gt_reweight = get_surface_reweighting_batch(gt_box_feature[:, 3:6], self.unit_cube.size(0))
        dist1, dist2 = self.chamferLoss(gt_box_pc, pred_box_pc)
        loss1 = (dist1 * gt_reweight).sum(dim=1) / (gt_reweight.sum(dim=1) + 1e-12)
        loss2 = (dist2 * pred_reweight).sum(dim=1) / (pred_reweight.sum(dim=1) + 1e-12)
        loss = (loss1 + loss2) / 2
        return loss

    def anchorLossEstimator(self, box_feature, gt_box_feature):
        pred_anchor_pc = transform_pc_batch(self.anchor, box_feature, anchor=True)
        gt_anchor_pc = transform_pc_batch(self.anchor, gt_box_feature, anchor=True)
        dist1, dist2 = self.chamferLoss(gt_anchor_pc, pred_anchor_pc)
        loss = (dist1.mean(dim=1) + dist2.mean(dim=1)) / 2
        return loss

    def isLeafLossEstimator(self, is_leaf_logit, gt_is_leaf):
        return self.bceLoss(is_leaf_logit, gt_is_leaf).view(-1)

    # decode a root code into a tree structure
    def decode_structure(self, z, max_depth):
        root_latent = self.sample_decoder(z)
        root = self.decode_node(root_latent, max_depth, Tree.root_sem)
        obj = Tree(root=root)
        return obj

    # decode a part node
    def decode_node(self, node_latent, max_depth, full_label, is_leaf=False):
        if node_latent.shape[0] != 1:
            raise ValueError('Node decoding does not support batch_size > 1.')

        is_leaf_logit = self.leaf_classifier(node_latent)
        node_is_leaf = is_leaf_logit.item() > 0

        # use maximum depth to avoid potential infinite recursion
        if max_depth < 1:
            is_leaf = True

        # decode the current part box
        box = self.box_decoder(node_latent)

        if node_is_leaf or is_leaf:
            ret = Tree.Node(is_leaf=True, full_label=full_label, label=full_label.split('/')[-1])
            ret.set_from_box_quat(box.view(-1))
            return ret
        else:
            child_feats, child_sem_logits, child_exists_logit, edge_exists_logits = \
                    self.child_decoder(node_latent)

            child_sem_logits = child_sem_logits.cpu().numpy().squeeze()

            # children
            child_nodes = []
            child_idx = {}
            for ci in range(child_feats.shape[1]):
                if torch.sigmoid(child_exists_logit[:, ci, :]).item() > 0.5:
                    idx = np.argmax(child_sem_logits[ci, Tree.part_name2cids[full_label]])
                    idx = Tree.part_name2cids[full_label][idx]
                    child_full_label = Tree.part_id2name[idx]
                    child_nodes.append(self.decode_node(\
                            child_feats[:, ci, :], max_depth-1, child_full_label, \
                            is_leaf=(child_full_label not in Tree.part_non_leaf_sem_names)))
                    child_idx[ci] = len(child_nodes) - 1

            # edges
            child_edges = []
            nz_inds = torch.nonzero(torch.sigmoid(edge_exists_logits) > 0.5)    
            edge_from = nz_inds[:, 1]
            edge_to = nz_inds[:, 2]
            edge_type = nz_inds[:, 3]

            for i in range(edge_from.numel()):
                cur_edge_from = edge_from[i].item()
                cur_edge_to = edge_to[i].item()
                cur_edge_type = edge_type[i].item()

                if cur_edge_from in child_idx and cur_edge_to in child_idx:
                    child_edges.append({
                        'part_a': child_idx[cur_edge_from],
                        'part_b': child_idx[cur_edge_to],
                        'type': self.conf.edge_types[cur_edge_type]})

            node = Tree.Node(is_leaf=False, children=child_nodes, edges=child_edges, \
                    full_label=full_label, label=full_label.split('/')[-1])
            node.set_from_box_quat(box.view(-1))
            return node

    # use gt structure, compute the reconstruction losses
    def structure_recon_loss(self, z, gt_tree):
        root_latent = self.sample_decoder(z)
        losses, _, _ = self.node_recon_loss(root_latent, gt_tree.root)
        return losses

    # compute per-node loss + children relationship loss
    def node_recon_loss(self, node_latent, gt_node):
        if gt_node.is_leaf:
            box = self.box_decoder(node_latent)
            box_loss = self.boxLossEstimator(box, gt_node.get_box_quat().view(1, -1))
            anchor_loss = self.anchorLossEstimator(box, gt_node.get_box_quat().view(1, -1))
            is_leaf_logit = self.leaf_classifier(node_latent)
            is_leaf_loss = self.isLeafLossEstimator(is_leaf_logit, is_leaf_logit.new_tensor(gt_node.is_leaf).view(1, -1))
            return {'box': box_loss, 'leaf': is_leaf_loss, 'anchor': anchor_loss, 
                    'exists': torch.zeros_like(box_loss), 'semantic': torch.zeros_like(box_loss),
                    'edge_exists': torch.zeros_like(box_loss),
                    'sym': torch.zeros_like(box_loss), 'adj': torch.zeros_like(box_loss)}, box, box
        else:
            child_feats, child_sem_logits, child_exists_logits, edge_exists_logits = \
                    self.child_decoder(node_latent)

            # generate box prediction for each child
            feature_len = node_latent.size(1)
            child_pred_boxes = self.box_decoder(child_feats.view(-1, feature_len))
            num_child_parts = child_pred_boxes.size(0)

            # perform hungarian matching between pred boxes and gt boxes
            with torch.no_grad():
                child_gt_boxes = torch.cat([child_node.get_box_quat().view(1, -1) for child_node in gt_node.children], dim=0)
                num_gt = child_gt_boxes.size(0)

                child_pred_boxes_tiled = child_pred_boxes.unsqueeze(dim=0).repeat(num_gt, 1, 1)
                child_gt_boxes_tiled = child_gt_boxes.unsqueeze(dim=1).repeat(1, num_child_parts, 1)

                dist_mat = self.boxLossEstimator(child_gt_boxes_tiled.view(-1, 10), child_pred_boxes_tiled.view(-1, 10)).view(-1, num_gt, num_child_parts)

                _, matched_gt_idx, matched_pred_idx = linear_assignment(dist_mat)

                # get edge ground truth
                edge_type_list_gt, edge_indices_gt = gt_node.edge_tensors(
                    edge_types=self.conf.edge_types, device=child_feats.device, type_onehot=False)

                gt2pred = {gt_idx: pred_idx for gt_idx, pred_idx in zip(matched_gt_idx, matched_pred_idx)}
                edge_exists_gt = torch.zeros_like(edge_exists_logits)

                sym_from = []; sym_to = []; sym_mat = []; sym_type = []; adj_from = []; adj_to = [];
                for i in range(edge_indices_gt.shape[1]//2):
                    if edge_indices_gt[0, i, 0].item() not in gt2pred or edge_indices_gt[0, i, 1].item() not in gt2pred:
                        """
                            one of the adjacent nodes of the current gt edge was not matched 
                            to any node in the prediction, ignore this edge
                        """
                        continue
                    
                    # correlate gt edges to pred edges
                    edge_from_idx = gt2pred[edge_indices_gt[0, i, 0].item()]
                    edge_to_idx = gt2pred[edge_indices_gt[0, i, 1].item()]
                    edge_exists_gt[:, edge_from_idx, edge_to_idx, edge_type_list_gt[0:1, i]] = 1
                    edge_exists_gt[:, edge_to_idx, edge_from_idx, edge_type_list_gt[0:1, i]] = 1

                    # compute binary edge parameters for each matched pred edge
                    if edge_type_list_gt[0, i].item() == 0: # ADJ
                        adj_from.append(edge_from_idx)
                        adj_to.append(edge_to_idx)
                    else:   # SYM
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
            is_leaf_logit = self.leaf_classifier(node_latent)
            is_leaf_loss = self.isLeafLossEstimator(is_leaf_logit, is_leaf_logit.new_tensor(gt_node.is_leaf).view(1, -1))

            # train the current node box to gt
            all_boxes = []; all_leaf_boxes = [];
            box = self.box_decoder(node_latent)
            all_boxes.append(box)
            box_loss = self.boxLossEstimator(box, gt_node.get_box_quat().view(1, -1))
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
            child_sem_gt_labels = torch.tensor(child_sem_gt_labels, dtype=torch.int64, \
                    device=child_sem_pred_logits.device)
            semantic_loss = self.semCELoss(child_sem_pred_logits, child_sem_gt_labels)
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
            child_exists_loss = F.binary_cross_entropy_with_logits(\
                input=child_exists_logits, target=child_exists_gt, reduction='none')
            child_exists_loss = child_exists_loss.sum()

            # train edge exists scores
            edge_exists_loss = F.binary_cross_entropy_with_logits(\
                    input=edge_exists_logits, target=edge_exists_gt, reduction='none')
            edge_exists_loss = edge_exists_loss.sum()
            # rescale to make it comparable to other losses, 
            # which are in the order of the number of child nodes
            edge_exists_loss = edge_exists_loss / (edge_exists_gt.shape[2]*edge_exists_gt.shape[3]) 

            # compute and train binary losses
            sym_loss = 0
            if len(sym_from) > 0:
                sym_from_th = torch.tensor(sym_from, dtype=torch.long, device=self.conf.device)
                obb_from = child_pred_boxes[sym_from_th, :]
                with torch.no_grad():
                    reweight_from = get_surface_reweighting_batch(obb_from[:, 3:6], self.unit_cube.size(0))
                pc_from = transform_pc_batch(self.unit_cube, obb_from)
                sym_to_th = torch.tensor(sym_to, dtype=torch.long, device=self.conf.device)
                obb_to = child_pred_boxes[sym_to_th, :]
                with torch.no_grad():
                    reweight_to = get_surface_reweighting_batch(obb_to[:, 3:6], self.unit_cube.size(0))
                pc_to = transform_pc_batch(self.unit_cube, obb_to)
                sym_mat_th = torch.cat(sym_mat, dim=0)
                transformed_pc_from = pc_from.matmul(torch.transpose(sym_mat_th[:, :, :3], 1, 2)) + \
                        sym_mat_th[:, :, 3].unsqueeze(dim=1).repeat(1, pc_from.size(1), 1)
                dist1, dist2 = self.chamferLoss(transformed_pc_from, pc_to)
                loss1 = (dist1 * reweight_from).sum(dim=1) / (reweight_from.sum(dim=1) + 1e-12)
                loss2 = (dist2 * reweight_to).sum(dim=1) / (reweight_to.sum(dim=1) + 1e-12)
                loss = loss1 + loss2
                sym_loss = loss.sum()

            adj_loss = 0
            if len(adj_from) > 0:
                adj_from_th = torch.tensor(adj_from, dtype=torch.long, device=self.conf.device)
                obb_from = child_pred_boxes[adj_from_th, :]
                pc_from = transform_pc_batch(self.unit_cube, obb_from)
                adj_to_th = torch.tensor(adj_to, dtype=torch.long, device=self.conf.device)
                obb_to = child_pred_boxes[adj_to_th, :]
                pc_to = transform_pc_batch(self.unit_cube, obb_to)
                dist1, dist2 = self.chamferLoss(pc_from, pc_to)
                loss = (dist1.min(dim=1)[0] + dist2.min(dim=1)[0])
                adj_loss = loss.sum()

            # call children + aggregate losses
            pred2allboxes = dict(); pred2allleafboxes = dict();
            for i in range(len(matched_gt_idx)):
                child_losses, child_all_boxes, child_all_leaf_boxes = self.node_recon_loss(
                    child_feats[:, matched_pred_idx[i], :], gt_node.children[matched_gt_idx[i]])
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
                sym_loss = sym_loss + child_losses['sym']
                adj_loss = adj_loss + child_losses['adj']

            # for sym-edges, train subtree to be symmetric
            for i in range(len(sym_from)):
                s1 = pred2allboxes[sym_from[i]].size(0)
                s2 = pred2allboxes[sym_to[i]].size(0)
                if s1 > 1 and s2 > 1:
                    obbs_from = pred2allboxes[sym_from[i]][1:, :]
                    obbs_to = pred2allboxes[sym_to[i]][1:, :]
                    pc_from = transform_pc_batch(self.unit_cube, obbs_from).view(-1, 3)
                    pc_to = transform_pc_batch(self.unit_cube, obbs_to).view(-1, 3)
                    transformed_pc_from = pc_from.matmul(torch.transpose(sym_mat[i][0, :, :3], 0, 1)) + \
                            sym_mat[i][0, :, 3].unsqueeze(dim=0).repeat(pc_from.size(0), 1)
                    dist1, dist2 = self.chamferLoss(transformed_pc_from.view(1, -1, 3), pc_to.view(1, -1, 3))
                    sym_loss += (dist1.mean() + dist2.mean()) * (s1 + s2) / 2

            # for adj-edges, train leaf-nodes in subtrees to be adjacent
            for i in range(len(adj_from)):
                if pred2allboxes[adj_from[i]].size(0) > pred2allleafboxes[adj_from[i]].size(0) \
                        or pred2allboxes[adj_to[i]].size(0) > pred2allleafboxes[adj_to[i]].size(0):
                    obbs_from = pred2allleafboxes[adj_from[i]]
                    obbs_to = pred2allleafboxes[adj_to[i]]
                    pc_from = transform_pc_batch(self.unit_cube, obbs_from).view(1, -1, 3)
                    pc_to = transform_pc_batch(self.unit_cube, obbs_to).view(1, -1, 3)
                    dist1, dist2 = self.chamferLoss(pc_from, pc_to)
                    adj_loss += dist1.min() + dist2.min()

            return {'box': box_loss + unused_box_loss, 'leaf': is_leaf_loss, 'anchor': anchor_loss, 
                    'exists': child_exists_loss, 'semantic': semantic_loss,
                    'edge_exists': edge_exists_loss, 'sym': sym_loss, 'adj': adj_loss}, \
                            torch.cat(all_boxes, dim=0), torch.cat(all_leaf_boxes, dim=0)

