from data_loader import DataLoader
from clustering import Clustering
import pandas as pd 
dl = DataLoader()
labeled_after_feature_df = dl.load_labeled_after_feature()
labeled_for_feature_df = dl.load_labeled_for_feature()
labeled_after_feature_x, labeled_after_feature_y = dl.get_xy(labeled_after_feature_df)


cluster = Clustering()
label = cluster.kmeans(labeled_after_feature_x)
temp = pd.concat([labeled_after_feature_df, label],axis=1)
print(temp)