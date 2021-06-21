import numpy as np
import math

class KNeighborsClassifier:

    def __init__(self,X_train,y_train,k): 
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
    # tính khoảng cách giữ 2 vector đặc trưng
    def calcDistancs(self,pointA, pointB):
        tmp = 0
        numOfFeature=12
        for i in range(numOfFeature):
            tmp += (float(pointA[i]) - float(pointB[i])) ** 2
        return math.sqrt(tmp)
    #Tìm ra k điểm gần nhất, trả về của hàm này là k nhãn gần nhất
    def kNearestNeighbor(self,X_train, y_train, point, k):
        distances = []
        #tính khoảng cách của vector đầu vào với từng vector trong dữ liệu
        for i in range(len(X_train)):
            distances.append({
                "label": y_train[i],
                "value": self.calcDistancs(X_train[i], point)
            })
        # sắp xếp tăng giần theo value(theo khoảng cách)
        distances.sort(key=lambda x: x["value"])
        #print(distances)
        labels = [item["label"] for item in distances]
        # lấy ra k nhãn có khoảng cách gần nhất
        return labels[:k]
    # tìm ra nhãn chiếm số lượng lớn nhất trong k nhãn tìm được
    def findMostOccur(self,arr):
        labels = set(arr) # lấy ra có bao nhiêu nhãn
        ans = ""
        maxOccur = 0
        for label in labels:
            num = arr.count(label)# đếm xem có bao nhiêu nhãn giống nhau
            if num > maxOccur:
                maxOccur = num
                ans = label
        # trả về nhãn chiếm số lượng lớn nhất 
        return ans
   # nhận dạng
    def predict(self,X_test):
        res = []
        for item in X_test:
            knn = self.kNearestNeighbor(self.X_train, self.y_train, item, self.k)
            res.append(self.findMostOccur(knn))
        return np.array(res)
