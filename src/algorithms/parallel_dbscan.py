import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN

class ParallelDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, n_jobs=4):
        self.eps = eps
        self.min_samples = min_samples
        self.n_jobs = n_jobs
        self.labels_ = None
        
    def fit_predict(self, X):
        dbscan = SklearnDBSCAN(
            eps=self.eps, 
            min_samples=self.min_samples, 
            n_jobs=self.n_jobs
        )
        self.labels_ = dbscan.fit_predict(X)
        return self.labels_
