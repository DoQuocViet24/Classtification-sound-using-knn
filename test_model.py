import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import extract_features
def classtification(filetest):
   # đọc dữ liệu đã được train
    features_df = pd.read_csv("train\\train.csv",header=0)
    Z = features_df["feature"].tolist()
    res = [i.strip("[]").split(", ") for i in Z]
    X = []
    for i in res:
        X.append([float(j) for j in i])
    X_train = np.array(X)    
    y_train = np.array(features_df["class_label"].tolist())
    # Trích rút đặc trưng dữ liệu âm thanh cần nhận dạng
    feature_test = extract_features.extract_features(filetest)
    X_test = np.array(feature_test.feature.tolist())
    # Bắt đầu nhận dạng 
    model = KNeighborsClassifier(n_neighbors=5)
    model = model.fit(X_train, y_train)
    result = model.predict(X_test)
    for i in result:
        if(i==0) :
            print("honda\n")
        elif(i==1):
            print("suzuki\n")
        elif(i==2):
            print("yamaha\n")
        else:
            print("không nhận ra\n")
def rate():
    features_df = pd.read_csv("train\\train.csv",header=0)
    y = np.array(features_df["class_label"].tolist())
    Z = features_df["feature"].tolist()
    res = [i.strip("[]").split(", ") for i in Z]
    X = []
    for i in res:
        X.append([float(j) for j in i])
    X = np.array(X)    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
    model = KNeighborsClassifier(n_neighbors=5)
    model = model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("Accuracy",acc*100,"%") 

if __name__ == "__main__":
    classtification("datatest")
    rate()


