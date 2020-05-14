import json
from collections import namedtuple
import numpy as np

def read_textfile_list(filename):
    object_names = []
    with open(filename) as f:
        object_names = f.readlines()
    object_names = [x.strip() for x in object_names]
    object_names = list(filter(lambda x: x is not None, object_names))
    #
    return object_names

def hex2rgb(h):
    return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def rgb2hex(rgb):
    return '#{0:02x}{1:02x}{2:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def load_pts(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        pts = np.array([[float(line.split()[0]), float(line.split()[1]), float(line.split()[2])] for line in lines], dtype=np.float32)
        return pts

def load_semantic_colors(filename):
    semantic_colors = {}
    with open(filename, 'r') as fin:
        for l in fin.readlines():
            semantic, r, g, b = l.rstrip().split()
            semantic_colors[semantic] = (int(r), int(g), int(b))

    return semantic_colors

def load_colors(filename):
    colors = []
    with open(filename, 'r') as fin:
        for l in fin.readlines():
            r, g, b = l.rstrip().split()
            colors.append((int(r), int(g), int(b)))

    return colors

def load_semantics_list(filename):
    semantics = []
    with open(filename, 'r') as fin:
        for l in fin.readlines():
            _, semantic, _ = l.rstrip().split()
            semantics.append(semantic)

    return semantics

class Tree(object):

    # global semantics
    part_name2id = None
    part_id2name = None
    part_name2cids = None
    part_non_leaf_sem_names = None
    num_sem = None

    @ staticmethod
    def load_semantics(filename):
        Tree.part_name2id = dict()
        Tree.part_id2name = dict()
        Tree.part_name2cids = dict()
        Tree.part_non_leaf_sem_names = []
        with open(filename, 'r') as fin:
            for l in fin.readlines():
                x, y, _ = l.rstrip().split()
                x = int(x)
                Tree.part_name2id[y] = x
                Tree.part_id2name[x] = y
                Tree.part_name2cids[y] = []
                if '/' in y:
                    Tree.part_name2cids['/'.join(y.split('/')[:-1])].append(x)
        Tree.num_sem = len(Tree.part_name2id) + 1
        for k in Tree.part_name2cids:
            Tree.part_name2cids[k] = np.array(Tree.part_name2cids[k], dtype=np.int32)
            if len(Tree.part_name2cids[k]) > 0:
                Tree.part_non_leaf_sem_names.append(k)

    class Node(object):
        def __init__(self, part_id=0, is_leaf=False, box=None, label=None, children=None, edges=None, hyperedges=None, full_label=None, geo=None, geo_feat=None):
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
            self.hyperedges = [] if hyperedges is None else hyperedges # hyperedges are an alternative to edges (multiple children are connected by hyperedges)
            """
                Hyperedge format:
                    parts:  the index into self.children for all children that are connected by the hyperedge
                    type:   ADJ, ROT_SYM, TRANS_SYM, REF_SYM
                    params: there is no params field for ADJ edge
                            for ROT_SYM edge, 0-2 pivot point, 3-5 axis unit direction, 6 radian rotation angle
                            for TRANS_SYM edge, 0-2 translation vector
                            for REF_SYM edge, 0-2 the middle point of the segment that connects the two box centers, 3-5 unit normal direction of the reflection plane
            """

        def get_semantic_id(self):
            return Tree.part_name2id[self.full_label]

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
            lbl = self.label if self.label is not None else '[no label]'
            out_str = '  |'*(level-1) + '  ├'*(level > 0) + str(pid) + ' ' + lbl + (' [LEAF] ' if self.is_leaf else '    ') + '{' + str(self.part_id) + '}'
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
                s = s + '{}{}:{}\n'.format(prefix, Tree.NodeType(node.type.item()).name, node_label)

                stack.extend(reversed(node.children))
                d_stack.extend([node_d+1]*len(node.children))

            return s

        def depth_first_traversal(self, get_depth=False):
            nodes = []
            if get_depth:
                depth = []

            stack = [self]
            while len(stack) > 0:
                node = stack.pop()
                nodes.append(node)

                stack.extend(reversed(node.children))

            if get_depth:
                return nodes, depth
            else:
                return nodes

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

        def sort_children(self, type):
            nodes = self.depth_first_traversal()

            for node in nodes:
                # sort first by label, then by 100z + 10y + x
                if type == 'label_xzy':
                    indices = list(range(len(node.children)))
                    indices.sort(key=lambda i: (node.children[i].label, node.children[i].box[0]*100 + node.children[i].box[2]*10 + node.children[i].box[1]*1 if node.children[i].box is not None else 0))
                    node.children = [node.children[i] for i in indices]
                    for edge in node.edges:
                        edge['part_a'] = indices.index(edge['part_a'])
                        edge['part_b'] = indices.index(edge['part_b'])
                    for hyperedge in node.hyperedges:
                        hyperedge['parts'] = [indices.index(p) for p in hyperedge['parts']]
                else:
                    raise ValueError('Unknown child sorting method {}.'.format(type))

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

    def sort_children(self, type):
        self.root.sort_children(type=type)

def load_object(fn, load_geo=False, load_geo_feat=False):
    #
    if load_geo:
        geo_fn = fn.replace('partnetobb', 'partnetgeo').replace('json', 'npz')
        geo_data = np.load(geo_fn)
    #
    if load_geo_feat:
        geo_fn = fn.replace('partnetobb', 'partnetgeofeat').replace('json', 'npz')
        geo_data = np.load(geo_fn)
    #
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
                node.geo = geo_data['parts'][node_json['id']].astype('float32').reshape(1, -1, 3)
            else:
                parent_id = parent.part_id
                other_idx = np.where(geo_data['other_ids'] == parent_id)[0][0]
                node.geo = geo_data['other_parts'][other_idx].astype('float32').reshape(1, -1, 3)

        if load_geo_feat:
            if node_json['id'] >= 0:
                node.geo_feat = geo_data['parts'][node_json['id']].astype('float32').reshape(1, -1)
            else:
                parent_id = parent.part_id
                other_idx = np.where(geo_data['other_ids'] == parent_id)[0][0]
                node.geo_feat = geo_data['other_parts'][other_idx].astype('float32').reshape(1, -1)

        if 'box' in node_json:
            node.box = np.array(node_json['box']).astype('float32')

        if 'children' in node_json:
            for ci, child in enumerate(node_json['children']):
                stack.append(StackElement(node_json=node_json['children'][ci], parent=node, parent_child_idx=ci))

        if 'edges' in node_json:
            for edge in node_json['edges']:
                if 'params' in edge:
                    edge['params'] = np.array(edge['params']).astype('float32')
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
