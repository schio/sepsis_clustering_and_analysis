from data_loader import DataLoader
from clustering import Clustering
from analysis import Analysis
import pandas as pd 


dl = DataLoader()
labeled_after_feature_df = dl.load_labeled_after_feature()
labeled_for_feature_df = dl.load_labeled_for_feature()

labeled_after_feature_x, labeled_after_feature_y = dl.get_xy(labeled_after_feature_df)
labeled_for_feature_x, labeled_for_feature_y = dl.get_xy(labeled_for_feature_df)

std_labeled_after_feature_x = dl.get_standard_scaler(labeled_after_feature_x)
std_labeled_for_feature_x = dl.get_standard_scaler(labeled_for_feature_x)


cluster = Clustering()
category_after_df = cluster.kmeans(labeled_after_feature_x, labeled_after_feature_df, "labeled_after_feature")
category_after_df = dl.ohe(category_after_df, ['cluster_category'])
category_for_df = cluster.kmeans(labeled_for_feature_x, labeled_for_feature_df, "labeled_for_feature")
category_for_df = dl.ohe(category_for_df, ['cluster_category'])

category_std_after_df = cluster.kmeans(labeled_after_feature_x, labeled_after_feature_df, "std_labeled_after_feature")
category_std_after_df = dl.ohe(category_std_after_df, ['cluster_category'])
category_std_for_df = cluster.kmeans(labeled_for_feature_x, labeled_for_feature_df, "std_labeled_for_feature")
category_std_for_df = dl.ohe(category_std_for_df, ['cluster_category'])

analysis = Analysis()
analysis.basic_characteristics(dl.get_xy(category_after_df)[0], labeled_after_feature_y, "cluster_labeled_after_feature")
analysis.basic_characteristics(dl.get_xy(category_for_df)[0], labeled_for_feature_y, "cluster_labeled_for_feature")
analysis.basic_characteristics(dl.get_xy(category_std_after_df)[0], labeled_after_feature_y, "cluster_std_labeled_after_feature")
analysis.basic_characteristics(dl.get_xy(category_std_for_df)[0], labeled_for_feature_y, "cluster_std_labeled_for_feature")
