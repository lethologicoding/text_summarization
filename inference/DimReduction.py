import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn import metrics

def run_umap(data = None):
    '''
    Runs umap dimensionality reduction on data
    '''    
    return umap.UMAP(random_state=42).fit_transform(data)

def get_optimal_k(df=pd.DataFrame, k_limit=30):
    '''
    Finds optimal k using max(silhouette_score) and being above .4
    '''
    coeffs=[]
    fewest_k = 2 # num of k to search with in search 
    for i in range(fewest_k, k_limit):
        best_score = 0
        cluster_algo = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
        cluster_algo.fit(df)
        labels = cluster_algo.labels_
        sil_coeff = metrics.silhouette_score(df, labels,metric='euclidean')
        coeffs.append(sil_coeff)
        if sil_coeff < .4: # needs further logic 
            break

    best_coeff = np.max(coeffs)
    best_k = np.argmax(coeffs)+fewest_k
    return best_k, best_coeff

def run_kmeans(k=int, data=None): 
    '''
    Runs kmeans clustering algo on k clusters
    '''
    clusters=KMeans(n_clusters=k)
    clusters.fit(data)
    labels = clusters.labels_
    sil_coeff = metrics.silhouette_score(data, labels,metric='euclidean')
    return labels
