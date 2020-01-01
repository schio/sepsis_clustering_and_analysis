from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd

class Clustering():
    def __init__(self):
        a=5

    def kmeans(self, x):
        kmeans = KMeans(n_clusters=4, random_state=75).fit(x.values)
        label = pd.DataFrame(kmeans.labels_, columns=['labels'])
        return label
    