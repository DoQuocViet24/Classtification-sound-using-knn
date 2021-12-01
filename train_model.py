import numpy as np
import extract_features

if __name__ == "__main__":
    
    #Trích xuất dữ liệu train
    print("Extracting features..")
    features_df = extract_features.extract_features('train')
    # lưu các đặc trưng nhận được và nhãn vào file train.csv
    features_df.to_csv('train\\train.csv',index = False, header=True)
    print("Successful training !")
