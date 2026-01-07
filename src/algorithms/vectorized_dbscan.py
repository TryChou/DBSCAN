import numpy as np
from collections import deque
from scipy.spatial import cKDTree

class VectorizedDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.tree_ = None
        
    def fit_predict(self, X):
        n_samples = len(X)
        self.labels_ = np.full(n_samples, -1)
        
        # 构建KD树
        self.tree_ = cKDTree(X)
        
        cluster_id = 0
        
        for i in range(n_samples):
            if self.labels_[i] != -1:
                continue
                
            # 使用KD树查询邻域
            distances, indices = self.tree_.query(
                X[i], 
                k=n_samples,
                distance_upper_bound=self.eps
            )
            neighbors = indices[distances < np.inf]
            
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
                continue
                
            self.labels_[i] = cluster_id
            seed_set = deque(neighbors)
            if i in seed_set:
                seed_set.remove(i)
            
            while seed_set:
                j = seed_set.popleft()
                
                if self.labels_[j] == -1:
                    self.labels_[j] = cluster_id
                    
                if self.labels_[j] != -1:
                    continue
                    
                self.labels_[j] = cluster_id
                
                # 查询j的邻域
                distances_j, indices_j = self.tree_.query(
                    X[j], 
                    k=n_samples,
                    distance_upper_bound=self.eps
                )
                j_neighbors = indices_j[distances_j < np.inf]
                
                if len(j_neighbors) >= self.min_samples:
                    for n in j_neighbors:
                        if self.labels_[n] == -1:
                            seed_set.append(n)
                            
            cluster_id += 1
            
        return self.labels_
