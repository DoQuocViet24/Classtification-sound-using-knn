import os
import librosa
import numpy as np
import glob
import pandas as pd

#hàm trích rút đặc trưng của 1 file âm thanh
def get_features(file_name):
    X, sample_rate = librosa.load(file_name, sr=None) #X: Âm thanh ở dạng chuỗi số
    #sr: Tốc độ lấy mẫu là số lượng mẫu âm thanh được truyền trong một giây
    # mfcc (mel-frequency cepstrum)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=12) 
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    return mfccs_scaled   # gia tri trung binh cua 12 thuoc tinh chua thong tin ve am sac

#hàm trích rút đặc trưng của toàn bộ dữ liệu âm thanh
def extract_features(path):

    # lấy đường dẫn đến dữ liệu train
    folders = os.listdir(path)
    folders.sort()
    # Lưu đặc trưng của từng file âm thanh
    features_list = []
    #Duyệt từng file dựa vào đường dẫn
    for numberOfFolder, folder in enumerate(folders):      #enumerate: liệt kê file bên trong  
        for fileName in glob.glob(os.path.join(path,folder,"*.wav")):
            try:
                print("Extraction file :"+fileName)
                splitFile = fileName.split("\\")   # ["path","folder","audio.wav"]
                mfccs = get_features(fileName)
                mfccs = mfccs.tolist()
            except Exception as e:
                print("Extraction error")
                continue
            features_list.append([mfccs,numberOfFolder,splitFile[2]])  # ["feature","class_label","file_name"]

    features_df = pd.DataFrame(features_list,columns = ['feature','class_label','file_name'])    # in ra file csv
    return features_df
    

