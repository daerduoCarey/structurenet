import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

def get_pc_center(pc):
    return np.mean(pc, axis=0)

def get_pc_scale(pc):
    return np.sqrt(np.max(np.sum((pc - np.mean(pc, axis=0))**2, axis=1)))

def get_pca_axes(pc):
    axes = PCA(n_components=3).fit(pc).components_
    return axes

def get_chamfer_distance(pc1, pc2):
    dist = cdist(pc1, pc2)
    error = np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0))
    scale = get_pc_scale(pc1) + get_pc_scale(pc2)
    return error / scale

