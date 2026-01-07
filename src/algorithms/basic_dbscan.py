import numpy as np
from collections import deque

class BasicDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        
    def fit_predict(self, X):
        n_samples = len(X)
        self.labels_ = np.full(n_samples, -1)
        cluster_id = 0
        
        for i in range(n_samples):
            if self.labels_[i] != -1:
                continue
                
            neighbors = []
            point = X[i]
            for j in range(n_samples):
                if np.sqrt(np.sum((point - X[j]) ** 2)) < self.eps:
                    neighbors.append(j)
            
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
                
                j_neighbors = []
                for k in range(n_samples):
                    if np.sqrt(np.sum((X[j] - X[k]) ** 2)) < self.eps:
                        j_neighbors.append(k)
                
                if len(j_neighbors) >= self.min_samples:
                    for n in j_neighbors:
                        if self.labels_[n] == -1:
                            seed_set.append(n)
                            
            cluster_id += 1
            
        return self.labels_
