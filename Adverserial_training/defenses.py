import numpy as np
import matplotlib.pyplot as plt
import torch

class KMeans:
    def __init__(self, k: int, max_iter: int = 20) -> None:
        self.num_clusters = k
        self.cluster_centers = None
        self.max_iter = max_iter

    def distance(self,p,X):
        return np.sqrt(np.sum((p-X)**2,axis=1))

    def fit(self, X):
        image_shape = X.shape
        X = np.array(X.reshape(1,-1).permute(1,0))
        n_samples,n_feautres = X.shape
    
        self.cluster_centers = np.random.rand(self.num_clusters,n_feautres)*5
    
        for _ in range(self.max_iter):
            commMap = self.predict(X)

            # getting no of clusters items and their sum and updating accordingly
            cluster_params = np.zeros(self.num_clusters)
            self.cluster_centers = np.zeros((self.num_clusters,n_feautres))

            for i in range(n_samples):
                cluster_id = commMap[i]
                cluster_params[cluster_id] += 1   #no of samples
                self.cluster_centers[cluster_id] += X[i,:]  # updating mean sum 

            for i in range(self.num_clusters):
                if cluster_params[i]==0: continue
                self.cluster_centers[i] = self.cluster_centers[i]/cluster_params[i]
                
                
        clustered_image = np.zeros_like(X)
        indexes_center = self.predict(X)
        for i in range(n_samples):
            clustered_image[i,:] = self.cluster_centers[indexes_center[i]]
        clustered_image = torch.tensor(clustered_image).permute(1,0).reshape(image_shape)
        
        
        return clustered_image
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predicts the index of the closest cluster center for each data point
        n_samples,_ = X.shape
        commMap = [0 for i in range(n_samples)]
        for i in range(n_samples):
            d = self.distance(X[i,:],self.cluster_centers)
            p = np.argmin(d)

            commMap[i] = p
        return commMap   