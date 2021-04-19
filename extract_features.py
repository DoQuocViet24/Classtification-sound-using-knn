import os
import librosa
import numpy as np
import glob
import pandas as pd

def get_features(file_name):
    X, sample_rate = librosa.load(file_name, sr=None)
    # mfcc (mel-frequency cepstrum)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate,n_mfcc=12)
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    return mfccs_scaled

def extract_features(filename):

    # path to dataset containing 10 subdirectories of .ogg files
    sub_dirs = os.listdir(filename)
    sub_dirs.sort()

    features_list = []
    for label, sub_dir in enumerate(sub_dirs):  
        for file_name in glob.glob(os.path.join(filename,sub_dir,"*.wav")):
            try:
                print("Extraction file :"+file_name)
                mfccs = get_features(file_name)
                mfccs = mfccs.tolist()
            except Exception as e:
                print("Extraction error")
                continue
            features_list.append([mfccs,label])

    features_df = pd.DataFrame(features_list,columns = ['feature','class_label'])
    #print(features_df)    
    return features_df
    

