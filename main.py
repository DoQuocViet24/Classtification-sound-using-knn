import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
import extract_features
import knn

if __name__ == "__main__":
    
    # extract features
    print("Extracting features..")
    features_df = extract_features.extract_features('dataset')
    knn.knn(features_df)
