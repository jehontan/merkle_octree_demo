from copy import copy
import hashlib
import numpy as np
import open3d as o3d
from enum import Enum
from itertools import takewhile

class DataPoint:
    def __init__(self, pos, data):
        self.pos = np.array(pos)
        self.data = data

class AbstractMerkleOctreeNode:
    def __init__(self, min_bounds, max_bounds, level, parent):
        self.min_bounds = np.array(min_bounds)
        self.max_bounds = np.array(max_bounds)
        self.center = (self.min_bounds + self.max_bounds)/2
        self.level = level
        self.parent = parent
        self.hash = None

    def insert(self, point:DataPoint):
        raise NotImplementedError

    def update_hash(self):
        raise NotImplementedError
    
    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, AbstractMerkleOctreeNode):
            return self.hash == __o.hash
        else:
            return False

class MerkleOctreeLeafNode(AbstractMerkleOctreeNode):
    def __init__(self, min_bounds, max_bounds, parent):
        super().__init__(min_bounds, max_bounds, level=0, parent=parent)
        self.data = None

    def insert(self, point:DataPoint):
        # TODO: implement merge rules here
        self.data = point.data
        self.update_hash()

    def update_hash(self):
        self.hash = hashlib.sha1(self.data).digest()

        if self.parent:
            self.parent.update_hash()

class MerkleOctreeInnerNode(AbstractMerkleOctreeNode):
    child_idx = np.arange(8, dtype=int).reshape((2,2,2))

    def __init__(self, min_bounds, max_bounds, level, parent):
        super().__init__(min_bounds, max_bounds, level, parent)
        self.children = [None, None, None, None, None, None, None, None]
        # self.children = [[[None, None], [None, None]],[[None, None],[None, None]]]
        # 0: negative, 1: positive for each axis

    def in_bounds(self, point:DataPoint):
        return np.all(point.pos >= self.min_bounds) and np.all(point.pos <= self.max_bounds)

    def insert(self, point:DataPoint):
        if not self.in_bounds(point):
            # point is outside this node
            return

        d = (point.pos - self.center)

        idx = (d >= 0).astype(int)
        
        if self.get_child(idx) is None:
            lb = np.zeros(3)
            ub = np.zeros(3)

            for i in range(3):
                if idx[i]: # positive range
                    lb[i] = self.center[i]
                    ub[i] = self.max_bounds[i]
                else:      # negative range
                    lb[i] = self.min_bounds[i]
                    ub[i] = self.center[i]
                    
            if self.level == 1:
                self.set_child(idx, MerkleOctreeLeafNode(min_bounds=lb, max_bounds=ub, parent=self))
            else:
                self.set_child(idx, MerkleOctreeInnerNode(min_bounds=lb, max_bounds=ub, level=self.level-1, parent=self))
        
        self.get_child(idx).insert(point)

    def update_hash(self):
        h = hashlib.sha1()
        for child in self.children:
            if child:
                h.update(child.hash)
        self.hash = h.digest()

        if self.parent:
            self.parent.update_hash()

    def get_child(self, idx) -> AbstractMerkleOctreeNode:
        idx_ = self.child_idx[idx[0]][idx[1]][idx[2]]
        return self.children[idx_]

    def set_child(self, idx, value):
        idx_ = self.child_idx[idx[0]][idx[1]][idx[2]]
        self.children[idx_] = value

    def __iter__(self):
        return MerkleOctreeIterator(self)

    @property
    def hash_tree(self):
        return HashTreeIterator(self)
            
class MerkleOctree(MerkleOctreeInnerNode):
    def __init__(self, min_bounds, max_bounds, max_depth):
        super().__init__(min_bounds, max_bounds, level=max_depth-1, parent=None)

    @staticmethod
    def create_from_point_cloud(pcd, max_depth, expand_by=0.1):
        min_bound = pcd.get_min_bound() - expand_by
        max_bound = pcd.get_max_bound() + expand_by
        octree = MerkleOctree(min_bound, max_bound, max_depth)

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        for i in range(points.shape[0]):
            datapoint = DataPoint(points[i], colors[i])
            octree.insert(datapoint)

        return octree

    def to_point_cloud(self):
        points = o3d.utility.Vector3dVector()
        colors = o3d.utility.Vector3dVector()
        for node in self:
            if isinstance(node, MerkleOctreeLeafNode):
                points.append(node.center)
                colors.append(node.data)

        pcd = o3d.geometry.PointCloud(points)
        pcd.colors = colors

        return pcd

class MerkleOctreeIterator: # Depth-First
    def __init__(self, tree:MerkleOctreeInnerNode):
        self.stack = [tree]

    def __next__(self):
        if len(self.stack) == 0:
            raise StopIteration

        node = self.stack.pop()

        if isinstance(node, MerkleOctreeInnerNode):
            for child in node.children:
                if child:
                    self.stack.append(child)

        return node

class HashTreeIterator: # Depth-First
    def __init__(self, tree:MerkleOctreeInnerNode):
        self.stack = [(0, 0, tree)]

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.stack) == 0:
            raise StopIteration

        parent_idx, idx, node = self.stack.pop()
        
        if node.level > 0:
            for i, child in enumerate(node.children):
                if child:
                    self.stack.append((idx, i, child))

        return node.level, parent_idx, idx, node.hash

    def to_list(self):
        return [h for h in self]

class HashTree:
    def __init__(self, hash=None, level=0, parent=None):
        self.hash = hash
        self.level = level
        self.parent = parent
        self.children = [None, None, None, None, None, None, None, None]

    def create_from_encoded(encoded):
        encoded = copy(encoded)
        max_level, parent_idx, idx, hash = encoded.pop(0)
        
        root = HashTree(hash, max_level)

        ancestors = [root] # stack

        for level, parent_idx, idx, hash in encoded:
            while level != max_level - len(ancestors):
                ancestors.pop()

            node = HashTree(hash, level, ancestors[-1])
            ancestors[-1].children[idx] = node
            ancestors.append(node)

        return root

    def __iter__(self):
        return HashTreeIterator(self)
    
    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, HashTree):
            return self.hash == __o.hash
        else:
            return False

class ChangeStatus(Enum):
    ADD_TO_LOCAL = 1    # remote copy does not exist locally
    ADD_TO_REMOTE = 2   # local copy does not exist remotely
    MERGE_BOTH = 3           # local/remote copies conflict, apply merge rules

def compare_hash_trees(local_tree:HashTree, remote_tree:HashTree):
    # BFS
    queue = [(list(), local_tree, remote_tree)] # path, local node, remote node
    diff_tree = []
    
    while len(queue) > 0:
        path, local, remote = queue.pop(0)
        
        # compare children
        for i in range(8):
            local_child = local.children[i]
            remote_child = remote.children[i]
            child_path = path + [i]

            if local_child == remote_child:
                # no difference, skip
                continue

            if not local_child:
                # branch does not exist locally
                diff_tree.append((child_path, ChangeStatus.ADD_TO_LOCAL))
            elif not remote_child:
                # branch does not exist remotely
                diff_tree.append((child_path, ChangeStatus.ADD_TO_REMOTE))
            else:
                # different versions
                if local_child.level == 0:
                    # comparing leaf nodes, no refinement possible, merge data
                    diff_tree.append((child_path, ChangeStatus.MERGE_BOTH))
                else:
                    # drill down to find changes
                    queue.append((child_path, local_child, remote_child))

    return diff_tree