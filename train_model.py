import numpy as np
import extract_features
import knn

if __name__ == "__main__":
    
    # extract features
    print("Extracting features..")
    features_df = extract_features.extract_features('dataset')
    features_df.to_csv('train\\train.csv',index = False, header=True)
    print("Successful training !")
