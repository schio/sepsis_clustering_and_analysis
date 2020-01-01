from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd

class Clustering():
    def __init__(self):
        a=5

    def kmeans(self, x, origin_df, feature_name_for_save):
        kmeans = KMeans(n_clusters=4, random_state=75).fit(x.values)
        category = pd.DataFrame(kmeans.labels_, columns=['cluster_category'])
        category_origin_df = pd.concat([origin_df, category],axis=1)

        category_origin_df.to_csv(f"./cluster_{feature_name_for_save}.csv")
        return category_origin_df
    