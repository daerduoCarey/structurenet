import sys
import os
from collections import namedtuple
import json
import torch
import numpy as np
from torch.utils import data
from pyquaternion import Quaternion
from sklearn.decomposition import PCA
from utils import one_hot, export_ply_with_label
import trimesh

# load Chair.txt meta-file
part_name2id = dict(); part_id2name = dict(); part_name2cids = dict();
with open('Chair.txt', 'r') as fin:
    for l in fin.readlines():
        x, y, _ = l.rstrip().split()
        x = int(x)
        part_name2id[y] = x
        part_id2name[x] = y
        part_name2cids[y] = []
        if '/' in y:
            part_name2cids['/'.join(y.split('/')[:-1])].append(x)
num_sem = len(part_name2id) + 1
part_non_leaf_sem_names = []
for k in part_name2cids:
    part_name2cids[k] = np.array(part_name2cids[k], dtype=np.int32)
    if len(part_name2cids[k]) > 0:
        part_non_leaf_sem_names.append(k)


class Tree(object):

    class Node(object):
        def __init__(self, part_id=0, is_leaf=False, box=None, label=None, children=None, edges=None, full_label=None, geo=None, geo_feat=None):
            self.is_leaf = is_leaf        # store T/F for leaf or non-leaf
            self.part_id = part_id        # part_id in result_after_merging.json (with a special id -1 denoting the unlabeled part if exists for each non-leaf node)
            self.box = box                # box feature vector for all nodes
            self.label = label            # node semantic label (with a special label 'other' denoting the unlabeled portion if exists for each non-leaf node)
            self.full_label = full_label
            self.children = [] if children is None else children  # all of its children nodes; each entry is a Node instance
            self.edges = [] if edges is None else edges           # all of its children relationships; each entry is a tuple <part_a, part_b, type, params, dist>
            self.geo = geo             # 1 x 1000 x 3 point cloud
            self.geo_feat = geo_feat    # 1 x feat_len
            self.pred_geo = None
            self.pred_geo_feat = None
            """
                Edge format:
                        part_a, part_b: the order in self.children (e.g. 0, 1, 2, 3, ...)
                                        This is an directional edge for A->B
                                        If an edge is commutative, you may need to manually specify a B->A edge
                                        For example, an ADJ edge is only shown A->B, there is no edge B->A
                        type:           ADJ, ROT_SYM, TRANS_SYM, REF_SYM
                        dist:           for ADJ edge, it's the closest distance between two parts
                                        for SYM edge, it's the chamfer distance after matching part B to part A
                        params:         there is no params field for ADJ edge
                                        for ROT_SYM edge, 0-2 pivot point, 3-5 axis unit direction, 6 radian rotation angle
                                        for TRANS_SYM edge, 0-2 translation vector
                                        for REF_SYM edge, 0-2 the middle point of the segment that connects the two box centers, 3-5 unit normal direction of the reflection plane
            """
        
        def get_semantic_id(self):
            return part_name2id[self.full_label]
            
        def get_semantic_one_hot(self):
            out = np.zeros((1, num_sem), dtype=np.float32)
            out[0, part_name2id[self.full_label]] = 1
            return torch.tensor(out, dtype=torch.float32).to(device=self.box.device)
            
        def get_box_quat(self):
            box = self.box.cpu().numpy().squeeze()
            center = box[:3]
            size = box[3:6]
            xdir = box[6:9]
            xdir /= np.linalg.norm(xdir)
            ydir = box[9:]
            ydir /= np.linalg.norm(ydir)
            zdir = np.cross(xdir, ydir)
            zdir /= np.linalg.norm(zdir)
            rotmat = np.vstack([xdir, ydir, zdir]).T
            q = Quaternion(matrix=rotmat)
            quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
            box_quat = np.hstack([center, size, quat]).astype(np.float32)
            return torch.from_numpy(box_quat).view(1, -1).to(device=self.box.device)

        def set_from_box_quat(self, box_quat):
            box_quat = box_quat.cpu().numpy().squeeze()
            center = box_quat[:3]
            size = box_quat[3:6]
            q = Quaternion(box_quat[6], box_quat[7], box_quat[8], box_quat[9])
            rotmat = q.rotation_matrix
            box = np.hstack([center, size, rotmat[:, 0].flatten(), rotmat[:, 1].flatten()]).astype(np.float32)
            self.box = torch.from_numpy(box).view(1, -1)

        def compute_box_from_geo(self):

            points = self.geo.cpu().numpy().reshape(-1, 3)
            try:
                to_origin, size = trimesh.bounds.oriented_bounds(obj=points, angle_digits=1)
                center = to_origin[:3, :3].transpose().dot(-to_origin[:3, 3])
                xdir = to_origin[0, :3]
                ydir = to_origin[1, :3]

            except scipy.spatial.qhull.QhullError:
                print('WARNING: falling back to PCA OBB computation since the more accurate minimum OBB computation failed.')
                center = points.mean(axis=0, keepdims=True)
                points = points - center
                center = center[0, :]
                pca = PCA()
                pca.fit(points)
                pcomps = pca.components_
                points_local = np.matmul(pcomps, points.transpose()).transpose()
                size = points_local.max(axis=0) - points_local.min(axis=0)
                xdir = pcomps[0, :]
                ydir = pcomps[1, :]
            
            box = np.hstack([center, size, xdir, ydir]).astype(np.float32)
            self.box = torch.tensor(box, dtype=torch.float32).view(1, -1)

        def to(self, device):
            if self.box is not None:
                self.box = self.box.to(device)
            for edge in self.edges:
                if 'params' in edge:
                    edge['params'].to(device)
            if self.geo is not None:
                self.geo = self.geo.to(device)
            if self.geo_feat is not None:
                self.geo_feat = self.geo_feat.to(device)
            if self.pred_geo is not None:
                self.pred_geo = self.pred_geo.to(device)
            if self.pred_geo_feat is not None:
                self.pred_geo_feat = self.pred_geo_feat.to(device)

            for child_node in self.children:
                child_node.to(device)

            return self

        def _to_str(self, level, pid, detailed=False):
            out_str = '  |'*(level-1) + '  ├'*(level > 0) + str(pid) + ' ' + self.label + (' [LEAF] ' if self.is_leaf else '    ') + '{' + str(self.part_id) + '}'
            if detailed:
                out_str += 'Box('+';'.join([str(item) for item in self.box.numpy()])+')\n'
            else:
                out_str += '\n'

            if len(self.children) > 0:
                for idx, child in enumerate(self.children):
                    out_str += child._to_str(level+1, idx)

            if detailed and len(self.edges) > 0:
                for edge in self.edges:
                    if 'params' in edge:
                        edge = edge.copy() # so the original parameters don't get changed
                        edge['params'] = edge['params'].cpu().numpy()
                    out_str += '  |'*(level) + '  ├' + 'Edge(' + str(edge) + ')\n'

            return out_str

        def __str__(self):
            return self._to_str(0, 0)

        def print_tree(self):
            d_stack = [0]
            stack = [self]
            s = '\n'
            while len(stack) > 0:
                node = stack.pop()
                node_d = d_stack.pop()
                node_label = node.label if (node.label is not None and len(node.label) > 0) else '<no label>'

                prefix = ''.join([' |']*(node_d-1)+[' ├']*(node_d > 0))
                s = s + f'{prefix}{Tree.NodeType(node.type.item()).name}:{node_label}\n'

                stack.extend(reversed(node.children))
                d_stack.extend([node_d+1]*len(node.children))

            return s

        def depth_first_traversal(self):
            nodes = []

            stack = [self]
            while len(stack) > 0:
                node = stack.pop()
                nodes.append(node)

                stack.extend(reversed(node.children))

            return nodes

        def child_adjacency(self, typed=False, max_children=None):
            if max_children is None:
                adj = torch.zeros(len(self.children), len(self.children))
            else:
                adj = torch.zeros(max_children, max_children)

            if typed:
                edge_types = ['ADJ', 'ROT_SYM', 'TRANS_SYM', 'REF_SYM']

            for edge in self.edges:
                if typed:
                    edge_type_index = edge_types.index(edge['type'])
                    adj[edge['part_a'], edge['part_b']] = edge_type_index
                    adj[edge['part_b'], edge['part_a']] = edge_type_index
                else:
                    adj[edge['part_a'], edge['part_b']] = 1
                    adj[edge['part_b'], edge['part_a']] = 1

            return adj

        def geos(self, leafs_only=True):
            nodes = list(self.depth_first_traversal())
            out_geos = []; out_nodes = [];
            for node in nodes:
                if not leafs_only or node.is_leaf:
                    out_geos.append(node.geo)
                    out_nodes.append(node)
            return out_geos, out_nodes

        def release_all_geos(self):
            nodes = list(self.depth_first_traversal())
            for node in nodes:
                node.geo = None
                node.geo_feat = None
                node.pred_geo = None
                node.pred_geo_feat = None

        def pred_geo_feats(self, leafs_only=True):
            nodes = list(self.depth_first_traversal())
            out_feats = []; out_nodes = [];
            for node in nodes:
                if (not leafs_only or node.is_leaf) and (node.pred_geo_feat is not None):
                    out_feats.append(node.pred_geo_feat)
                    out_nodes.append(node)
            return out_feats, out_nodes

        def gt_and_pred_geos(self, leafs_only=True):
            nodes = list(self.depth_first_traversal())
            gt_geos = []; pred_geos = [];
            for node in nodes:
                if (not leafs_only or node.is_leaf) and (node.pred_geo is not None) and (node.geo is not None):
                    gt_geos.append(node.geo)
                    pred_geos.append(node.pred_geo)
            return gt_geos, pred_geos

        # compute the full set of boxes (inlcuding those given implicitly by symmetries)
        # symmetries nodes produce symmetric instances for all boxes in their subtree
        def boxes(self, per_node=False, leafs_only=False):

            nodes = list(reversed(self.depth_first_traversal()))
            node_boxesets = []
            boxes_stack = []
            for node in nodes:

                # children are on the stack right (top-most) to left (bottom-most)
                node_boxes = []
                for i in range(len(node.children)):
                    node_boxes = boxes_stack.pop() + node_boxes

                if node.box is not None and (not leafs_only or node.is_leaf):
                    node_boxes.append(node.box)

                # if node.sym is not None:
                #     node_boxes = sym_instances(boxes=node_boxes, s=node.sym)

                if per_node:
                    node_boxesets.append(node_boxes)

                boxes_stack.append(node_boxes)

            assert len(boxes_stack) == 1

            if per_node:
                return node_boxesets, list(nodes)
            else:
                boxes = boxes_stack[0]
                return boxes

        def graph(self, leafs_only=False):

            boxes = []
            edges = []
            box_ids = []

            nodes = list(reversed(self.depth_first_traversal()))

            box_index_offset = 0
            for node in nodes:

                child_count = 0
                box_idx = {}
                for i, child in enumerate(node.children):
                    if leafs_only and not child.is_leaf:
                        continue
                    boxes.append(child.box)
                    box_ids.append(child.part_id)
                    box_idx[i] = child_count+box_index_offset
                    child_count += 1

                for edge in node.edges:
                    if leafs_only and not (
                            node.children[edge['part_a']].is_leaf and
                            node.children[edge['part_b']].is_leaf):
                        continue
                    edges.append(edge.copy())
                    edges[-1]['part_a'] = box_idx[edges[-1]['part_a']]
                    edges[-1]['part_b'] = box_idx[edges[-1]['part_b']]

                box_index_offset += child_count

            return boxes, edges, box_ids

        """ Return:
                    edge_type:      1 x (NumEdgesx2) x NumTypes if type_onehot else 1 x (NumEdgesx2)
                    edge_feats:     1 x (NumEdgesx2) x FeatSize if feat_type == 'full_params' or None
                    edge_indices:   1 x (NumEdgesx2) x 2
        """
        def edge_tensors(self, feat_type, feat_size, edge_types, device, type_onehot=True):

            num_edges = len(self.edges)

            # get directed edge indices in both directions as tensor
            edge_indices = torch.tensor(
                [[e['part_a'], e['part_b']] for e in self.edges] + [[e['part_b'], e['part_a']] for e in self.edges],
                device=device, dtype=torch.long).view(1, num_edges*2, 2)

            if feat_type not in ['type_only', 'full_params']:
                raise ValueError(f'Unknown edge feature type: {feat_type}.')

            # get edge type as tensor
            edge_type = torch.tensor([edge_types.index(edge['type']) for edge in self.edges], device=device, dtype=torch.long)
            if type_onehot:
                edge_type = one_hot(inp=edge_type, label_count=len(edge_types)).transpose(0, 1).view(1, num_edges, len(edge_types)).to(dtype=torch.float32)
            else:
                edge_type = edge_type.view(1, num_edges)
            edge_type = torch.cat([edge_type, edge_type], dim=1) # add edges in other direction (symmetric adjacency)

            # get edge features (parameters) as tensor
            if feat_type in ['full_params']:
                num_edges = len(self.edges)
                edge_feats = torch.zeros((1, num_edges*2, 7), device=device, dtype=torch.float32)
                for ei, edge in enumerate(self.edges):
                    if 'params' in edge:
                        edge_feats[0, ei, :edge['params'].size(0)] = edge['params']
            else:
                edge_feats = None

            return edge_type, edge_feats, edge_indices

        def get_subtree_edge_count(self):
            cnt = 0
            if self.children is not None:
                for cnode in self.children:
                    cnt += cnode.get_subtree_edge_count()
            if self.edges is not None:
                cnt += len(self.edges)
            return cnt

        def sort_by_label(self):
            if self.children is not None:
                self.children.sort(key=lambda child: child.label)
                for cnode in self.children:
                    cnode.sort_by_label()

        # TODO: graph edit distance with one of these:
        # https://networkx.github.io/documentation/latest/reference/algorithms/generated/networkx.algorithms.similarity.graph_edit_distance.html#networkx.algorithms.similarity.graph_edit_distance
        # https://github.com/Jacobe2169/GMatch4py
        @staticmethod
        def distance(node1, node2, detailed=False, weights=None):

            if weights is not None:
                edit_dist_def = Tree.EditDistanceDefinition(
                    w_delete=weights['delete'],
                    w_insert=weights['insert'],
                    w_rename_label=weights['rename_label'],
                    w_rename_box=weights['rename_box'])
            else:
                edit_dist_def = Tree.EditDistanceDefinition()

            apted_tree = apted.APTED(node1, node2, edit_dist_def)
            edit_dist = apted_tree.compute_edit_distance()

            if detailed:
                mapping = apted_tree.compute_edit_mapping()

                rename_dist = 0.0
                delete_dist = 0.0
                insert_dist = 0.0

                for n1, n2 in mapping:
                    if n2 is None:
                        delete_dist += edit_dist_def.delete(n1)
                    elif n1 is None:
                        insert_dist += edit_dist_def.insert(n2)
                    else:
                        rename_dist += edit_dist_def.rename(n1, n2)

                return edit_dist, rename_dist, delete_dist, insert_dist

            else:
                return edit_dist

    class EditDistanceDefinition(apted.Config):

        def __init__(self, w_delete=2.0, w_insert=2.0, w_rename_label=100.0, w_rename_box=1.0):
            self.w_delete = w_delete
            self.w_insert = w_insert
            self.w_rename_label = w_rename_label
            self.w_rename_box = w_rename_box

        # cost for deleting nodes
        def delete(self, node):
            return self.w_delete

        # cost for inserting nodes
        def insert(self, node):
            return self.w_insert

        # node distance - models are assumed to be normalized!
        # otherwise the box and sym losses might get larger
        def rename(self, node1, node2):
            # type (100 if they are not the same type)
            dist = int(node1.label != node2.label) * self.w_rename_label

            # box
            if node1.box is not None and node2.box is not None:
                dist += (node1.box.cpu() - node2.box.cpu()).pow_(2).mean(dim=1).item() * self.w_rename_box

            return dist

        # get children of a node
        def children(self, node):
            return node.children


    def __init__(self, root):
        self.root = root

    def to(self, device):
        self.root = self.root.to(device)
        return self

    def __str__(self):
        return str(self.root)

    def print_tree(self):
        return self.root.print_tree()

    def depth_first_traversal(self):
        return self.root.depth_first_traversal()

    def boxes(self, per_node=False, leafs_only=False):
        return self.root.boxes(per_node=per_node, leafs_only=leafs_only)

    def graph(self, leafs_only=False):
        return self.root.graph(leafs_only=leafs_only)

    def free(self):
        for node in self.depth_first_traversal():
            del node.geo
            del node.box
            del node.geo_feat
            del node.pred_geo_feat
            del node.pred_geo
            del node

    def sort_by_label(self):
        self.root.sort_by_label()

    @staticmethod
    def distance(tree1, tree2, detailed=False, weights=None):
        return Tree.Node.distance(node1=tree1.root, node2=tree2.root, detailed=detailed, weights=weights)


class PartNetObbDataset(data.Dataset):
    def __init__(self, root, object_list, data_features, load_geo=False, load_geo_feat=False):
        self.root = root
        self.data_features = data_features
        self.load_geo = load_geo
        self.load_geo_feat = load_geo_feat

        if isinstance(object_list, str):
            with open(object_list, 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]
        else:
            self.object_names = object_list

    def __getitem__(self, index):
        if 'object' in self.data_features:
            obj = self.load_object(os.path.join(self.root, self.object_names[index]+'.json'), \
                    load_geo=self.load_geo, load_geo_feat=self.load_geo_feat)

        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'name':
                data_feats = data_feats + (self.object_names[index],)
            else:
                assert False, 'ERROR: unknow feat type %s!' % feat
                
        return data_feats

    def __len__(self):
        return len(self.object_names)

    def get_anno_id(self, anno_id):
        obj = self.load_object(os.path.join(self.root, anno_id+'.json'), \
                load_geo=self.load_geo, load_geo_feat=self.load_geo_feat)
        return obj

    @staticmethod
    def load_object(fn, load_geo=False, load_geo_feat=False):

        if load_geo:
            geo_fn = fn.replace('partnetobb', 'partnetgeo').replace('json', 'npz')
            geo_data = np.load(geo_fn)

        if load_geo_feat:
            geo_fn = fn.replace('partnetobb', 'partnetgeofeat').replace('json', 'npz')
            geo_data = np.load(geo_fn)

        with open(fn, 'r') as f:
            root_json = json.load(f)

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node_json', 'parent', 'parent_child_idx'])
        stack = [StackElement(node_json=root_json, parent=None, parent_child_idx=None)]

        root = None
        # traverse the tree, converting each node json to a Node instance
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent = stack_elm.parent
            parent_child_idx = stack_elm.parent_child_idx
            node_json = stack_elm.node_json

            node = Tree.Node(
                part_id=node_json['id'],
                is_leaf=('children' not in node_json),
                label=node_json['label'])

            if load_geo:
                if node_json['id'] >= 0:
                    node.geo = torch.tensor(geo_data['parts'][node_json['id']], dtype=torch.float32).view(1, -1, 3)
                else:
                    parent_id = parent.part_id
                    other_idx = np.where(geo_data['other_ids'] == parent_id)[0][0]
                    node.geo = torch.tensor(geo_data['other_parts'][other_idx], dtype=torch.float32).view(1, -1, 3)

            if load_geo_feat:
                if node_json['id'] >= 0:
                    node.geo_feat = torch.tensor(geo_data['parts'][node_json['id']], dtype=torch.float32).view(1, -1)
                else:
                    parent_id = parent.part_id
                    other_idx = np.where(geo_data['other_ids'] == parent_id)[0][0]
                    node.geo_feat = torch.tensor(geo_data['other_parts'][other_idx], dtype=torch.float32).view(1, -1)

            if 'box' in node_json:
                node.box = torch.from_numpy(np.array(node_json['box'])).to(dtype=torch.float32)

            if 'children' in node_json:
                for ci, child in enumerate(node_json['children']):
                    stack.append(StackElement(node_json=node_json['children'][ci], parent=node, parent_child_idx=ci))

            if 'edges' in node_json:
                for edge in node_json['edges']:
                    if 'params' in edge:
                        edge['params'] = torch.from_numpy(np.array(edge['params'])).to(dtype=torch.float32)
                    node.edges.append(edge)

            if parent is None:
                root = node
                root.full_label = root.label
            else:
                if len(parent.children) <= parent_child_idx:
                    parent.children.extend([None] * (parent_child_idx+1-len(parent.children)))
                parent.children[parent_child_idx] = node
                node.full_label = parent.full_label + '/' + node.label

        obj = Tree(root=root)

        return obj

    @staticmethod
    def save_object(obj, fn):

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node', 'parent_json', 'parent_child_idx'])
        stack = [StackElement(node=obj.root, parent_json=None, parent_child_idx=None)]

        obj_json = None
        pc = []; label = []; label_id = 0;

        # traverse the tree, converting child nodes of each node to json
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent_json = stack_elm.parent_json
            parent_child_idx = stack_elm.parent_child_idx
            node = stack_elm.node

            node_json = {
                'id': node.part_id,
                'label': f'{node.label if node.label is not None else ""}'}

            if node.geo is not None and node.is_leaf:
                cur_pc = node.geo.cpu().numpy().reshape(-1, 3)
                n_point = cur_pc.shape[0]
                cur_label = np.ones((n_point), dtype=np.int32) * label_id
                pc.append(cur_pc)
                label.append(cur_label)
                label_id += 1

            if node.geo_feat is not None and node.is_leaf:
                cur_feat = node.geo_feat.cpu().numpy().reshape(-1)
                node_json['geo_feat'] = cur_feat.tolist()

            if node.box is not None:
                node_json['box'] = node.box.cpu().numpy().reshape(-1).tolist()

            if len(node.children) > 0:
                node_json['children'] = []
            for child in node.children:
                node_json['children'].append(None)
                stack.append(StackElement(node=child, parent_json=node_json, parent_child_idx=len(node_json['children'])-1))

            if len(node.edges) > 0:
                node_json['edges'] = []
            for edge in node.edges:
                node_json['edges'].append(edge)
                if 'params' in edge:
                    node_json['edges'][-1]['params'] = node_json['edges'][-1]['params'].cpu().numpy().reshape(-1).tolist()

            if parent_json is None:
                obj_json = node_json
            else:
                parent_json['children'][parent_child_idx] = node_json

        with open(fn, 'w') as f:
            json.dump(obj_json, f)

        if len(pc) > 0:
            pc = np.vstack(pc)
            label = np.hstack(label)
            export_ply_with_label(fn.replace('.json', '.ply'), pc, label)

    @staticmethod
    def get_child_graph(objects, label, categories, max_node_num, use_labels, skip_single_child=True):
        node_num = []
        boxes = torch.zeros(len(objects), 12, max_node_num)
        adj = torch.zeros(len(objects), max_node_num, max_node_num)
        if use_labels:
            labels = torch.zeros(len(objects), len(categories), max_node_num) # one-hot label encoding
            label_inds = torch.zeros(len(objects), max_node_num, dtype=torch.long) # one-hot label encoding
        else:
            labels = None
            label_inds = None
        for oi, obj in enumerate(objects):

            parent_node = None
            for node in obj.depth_first_traversal():
                if node.label == label:
                    parent_node = node
                    break

            if parent_node is None:
                raise ValueError(f'An object did not have the label {label}.')

            if skip_single_child and len(parent_node.children) == 1:
                parent_node = parent_node.children[0]

            child_nodes = parent_node.children

            if len(child_nodes) > max_node_num:
                print(f'WARNING: too many children for an object, removing additional children')
                child_nodes = child_nodes[:max_node_num]

            node_num.append(len(child_nodes))

            adj_size = min(max_node_num, len(parent_node.children))
            adj[oi, :adj_size, :adj_size] = parent_node.child_adjacency(typed=False)[:adj_size, :adj_size]

            for ci, child in enumerate(child_nodes):
                boxes[oi, :, ci] = child.box.view(-1)
                if use_labels:
                    labels[oi, categories.index(child.label), ci] = 1
                    label_inds[oi, ci] = categories.index(child.label)

        return boxes, labels, label_inds, adj, node_num

def collate_feats(b):
    return list(zip(*b))

