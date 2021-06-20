import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import extract_features

#hàm nhận diện âm thanh tiếng động cơ
def classtification(filetest):
   # đọc dữ liệu đã được train(X_train: feature, y_train: label)
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
    file = np.array(feature_test.file_name.tolist())

    # Bắt đầu nhận dạng 
    model = KNeighborsClassifier(n_neighbors=5)
    model = model.fit(X_train, y_train)
    result = model.predict(X_test)
    # nhận dạng và trả về nhãn của âm thanh đầu vào
    for i in range(len(result)):
        if(result[i]==0) :
            print(file[i]+ ": honda\n")
        elif(result[i]==1):
            print(file[i]+ ": suzuki\n")
        elif(result[i]==2):
            print(file[i]+ ": yamaha\n")
        else:
            print("không nhận ra\n")

# hàm đánh giá thuật toán knn
def rate():
    # đọc dữ liệu đã train (X: feature, y: label)
    features_df = pd.read_csv("train\\train.csv",header=0)
    Z = features_df["feature"].tolist()
    res = [i.strip("[]").split(", ") for i in Z]
    X = []
    for i in res:
        X.append([float(j) for j in i])
    X = np.array(X)    
    y = np.array(features_df["class_label"].tolist())
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
    model = KNeighborsClassifier(n_neighbors=5)
    model = model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("Accuracy",acc*100,"%") 

# hàm main
if __name__ == "__main__":
    classtification("datatest")
    rate()


