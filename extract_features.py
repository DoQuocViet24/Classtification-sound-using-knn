import os
import librosa
import numpy as np
import glob
import pandas as pd

#hàm trích rút đặc trưng của 1 file âm thanh
def get_features(file_name):
    X, sample_rate = librosa.load(file_name, sr=None)
    # mfcc (mel-frequency cepstrum)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate,n_mfcc=12) 
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    return mfccs_scaled

#hàm trích rút đặc trưng của toàn bộ dữ liệu âm thanh
def extract_features(filename):

    # lấy đường dẫn đến dữ liệu train
    sub_dirs = os.listdir(filename)
    sub_dirs.sort()
    # Lưu đặc trưng của từng file âm thanh
    features_list = []
    #Duyệt từng file dựa vào đường dẫn
    for label, sub_dir in enumerate(sub_dirs):  
        for file_name in glob.glob(os.path.join(filename,sub_dir,"*.wav")):
            try:
                print("Extraction file :"+file_name)
                file = file_name.split("\\")
                mfccs = get_features(file_name)
                mfccs = mfccs.tolist()
            except Exception as e:
                print("Extraction error")
                continue
            features_list.append([mfccs,label,file[2]])

    features_df = pd.DataFrame(features_list,columns = ['feature','class_label','file_name'])
    return features_df
    

