from data_loader import DataLoader
from clustering import Clustering
from analysis import Analysis
import pandas as pd 
import sys

def get_kmeans_filename(k, after, std):
    filename = f"{k}means"
    filename += "_std_labeled" if std else "_labeled"
    filename += "_after_feature.csv" if after else "_for_feature.csv"
    return filename

# each row data load -> preprocessing -> save
dl = DataLoader()
labeled_after_feature_df = dl.load_labeled_after_feature()
labeled_for_feature_df = dl.load_labeled_for_feature()

labeled_after_feature_x, labeled_after_feature_y = dl.get_xy(labeled_after_feature_df)
labeled_for_feature_x, labeled_for_feature_y = dl.get_xy(labeled_for_feature_df)
std_labeled_after_feature_x = dl.get_standard_scaler(labeled_after_feature_x)
std_labeled_for_feature_x = dl.get_standard_scaler(labeled_for_feature_x)

195438


# labeled data load -> labeling clustering result -> save
k = sys.argv[1]
cluster = Clustering(4)
category_after_df = cluster.kmeans(labeled_after_feature_x, labeled_after_feature_df, "labeled_after_feature")
category_after_df = dl.ohe(category_after_df, ['cluster_category'])
category_for_df = cluster.kmeans(labeled_for_feature_x, labeled_for_feature_df, "labeled_for_feature")
category_for_df = dl.ohe(category_for_df, ['cluster_category'])
category_std_after_df = cluster.kmeans(labeled_after_feature_x, labeled_after_feature_df, "std_labeled_after_feature")
category_std_after_df = dl.ohe(category_std_after_df, ['cluster_category'])
category_std_for_df = cluster.kmeans(labeled_for_feature_x, labeled_for_feature_df, "std_labeled_for_feature")
category_std_for_df = dl.ohe(category_std_for_df, ['cluster_category'])

category_after_df[category_after_df] 
category_for_df[category_for_df] 
category_std_after_df[category_std_after_df]
category_std_for_df[category_std_for_df]



df = df[df.line_race != 0]
print(category_std_for_df[category_std_for_df.cluster_category_3==1])
asd
# cluster labeled data load -> run basic_characteristics -> save
analysis = Analysis()
analysis.basic_characteristics(dl.get_xy(category_after_df)[0], labeled_after_feature_y, get_kmeans_filename(k, after=True, std=False))
analysis.basic_characteristics(dl.get_xy(category_for_df)[0], labeled_for_feature_y, get_kmeans_filename(k, after=False, std=False))
analysis.basic_characteristics(dl.get_xy(category_std_after_df)[0], labeled_after_feature_y, get_kmeans_filename(k, after=True, std=True))
analysis.basic_characteristics(dl.get_xy(category_std_for_df)[0], labeled_for_feature_y, get_kmeans_filename(k, after=False, std=True))



    