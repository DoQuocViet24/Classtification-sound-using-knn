import sklearn
from Knn import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import extract_features

#hàm nhận diện âm thanh tiếng con vật
def classification(path):
   # đọc dữ liệu đã có từ file train.csv(X_train: feature, y_train: label)
    features_df = pd.read_csv("train\\train.csv",header=0)
    arrayFeatures = features_df["feature"].tolist()
    res = [i.strip("[]").split(", ") for i in arrayFeatures]  #chuyển đổi dữ liệu về mảng có thể xử lý được
    X = []
    for i in res:
        X.append([float(j) for j in i])     # chuyển mảng giá trị về kiểu float
    X_train = np.array(X)    
    y_train = np.array(features_df["class_label"].tolist())     # lấy các nhãn

    # Trích rút đặc trưng của âm thanh đầu vào
    feature_test = extract_features.extract_features(path)
    X_test = np.array(feature_test.feature.tolist())
    fileName = np.array(feature_test.file_name.tolist())

    # Bắt đầu nhận dạng 
    model = KNeighborsClassifier(X_train,y_train,5)
    result = model.predict(X_test)
    # nhận dạng và trả về nhãn của âm thanh đầu vào
    for i in range(len(result)): 
        if(result[i]==0) :
            print(fileName[i]+ ": cat\n")
        elif(result[i]==1):
            print(fileName[i]+ ": dog\n")
        elif(result[i]==2):
            print(fileName[i]+ ": duck\n")
        elif(result[i]==3):
            print(fileName[i]+ ": pig\n")

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
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 27, random_state = 42)
    model = knn(n_neighbors=12)
    model = model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("Accuracy",acc*100,"%") 

# hàm main
if __name__ == "__main__":
    classification("test")
    rate()
    


