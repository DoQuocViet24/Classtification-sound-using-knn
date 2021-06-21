import numpy as np
import math

class KNeighborsClassifier:

    def __init__(self,X_train,y_train,k):   #khai bao constructor
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
    # tính khoảng cách giữ 2 vector đặc trưng
    def calDistances(self,pointA, pointB):
        distance = 0
        numOfFeature=12
        for i in range(numOfFeature):
            distance += (float(pointA[i]) - float(pointB[i])) ** 2
        return math.sqrt(distance)

    #Tìm ra k điểm gần nhất, trả về của hàm này là k nhãn gần nhất
    def kNearestNeighbor(self,X_train, y_train, pointTest, k):
        distances = []
        #tính khoảng cách của vector đầu vào với từng vector trong dữ liệu
        for i in range(len(X_train)):
            distances.append({
                "label": y_train[i],
                "value": self.calDistances(X_train[i], pointTest)
            })
        # sắp xếp tăng giần theo value(theo khoảng cách)
        distances.sort(key=lambda x: x["value"])
        #print(distances)
        labels = [item["label"] for item in distances]
        # lấy ra k nhãn có khoảng cách gần nhất
        return labels[:k]

    # tìm ra nhãn chiếm số lượng lớn nhất trong k nhãn tìm được
    def findMostLabels(self,labels):
        lb = set(labels) # lấy ra có bao nhiêu nhãn
        a = ""
        max = 0
        for i in lb:
            num = labels.count(i)# đếm xem có bao nhiêu nhãn giống nhau
            if  num > max:
                max = num
                a = i
        # trả về nhãn chiếm số lượng lớn nhất 
        return a
   # nhận dạng
    def predict(self,X_test):
        res = []
        for pointTest in X_test:
            labels = self.kNearestNeighbor(self.X_train, self.y_train, pointTest, self.k)
            res.append(self.findMostLabels(labels))
        return np.array(res)
