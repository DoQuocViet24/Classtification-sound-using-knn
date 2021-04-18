import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split

import extract_features
def knn(features_df):
    X = np.array(features_df.feature.tolist())
    y = np.array(features_df.class_label.tolist())
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
    # feature_test = extract_features.extract_features('datatest')
    # X_test = np.array(feature_test.feature.tolist())
    model = KNeighborsClassifier(n_neighbors=5)
    model = model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("Accuracy",acc*100,"%") 
    # result = model.predict(X_test)
    # for i in result:
    #     if(i==0) :
    #         print("honda\n")
    #     elif(i==1):
    #         print("suzuki\n")
    #     elif(i==2):
    #         print("yamaha\n")
    #     else:
    #         print("không nhận ra\n")

