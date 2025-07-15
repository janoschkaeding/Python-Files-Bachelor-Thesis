# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 11:22:23 2025

@author: Janos
"""

import h5py
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns 
import re


#Selecting the folders containing the h5 files
folders = [
    r"C:\Users\Janos\Desktop\blood\BSNP_dinobloom\dinobloom-c2f7990b-dab38abe",
    r"C:\Users\Janos\Desktop\blood\control_dinobloom\dinobloom-c2f7990b-dab38abe",
    r"C:\Users\Janos\Desktop\blood\NP_HS_dinobloom\dinobloom-c2f7990b-dab38abe",
    r"C:\Users\Janos\Desktop\blood\SH_dinobloom\dinobloom-c2f7990b-dab38abe",
    r"C:\Users\Janos\Desktop\blood\extra_crenatedcells_dinobloom\dinobloom-c2f7990b-dab38abe",
    r"C:\Users\Janos\Desktop\blood\extra_echinocyte_dinobloom\dinobloom-c2f7990b-dab38abe",
    r"C:\Users\Janos\Desktop\blood\extra_ecocentric_dinobloom\dinobloom-c2f7990b-dab38abe",
    r"C:\Users\Janos\Desktop\blood\extra_rouleaux_dinobloom\dinobloom-c2f7990b-dab38abe",
    r"C:\Users\Janos\Desktop\blood\extra_schistocyte_dinobloom\dinobloom-c2f7990b-dab38abe",
    r"C:\Users\Janos\Desktop\blood\extra_teardrop_dinobloom\dinobloom-c2f7990b-dab38abe",
]

#Setting the number of features to read from each file
feat_len = 384

#Creating the list of h5 files and 
h5_files = []
for folder in folders:
    files = glob.glob(os.path.join(folder, "*.h5"))
    files.sort(key=lambda path: int(re.search(r'(\d+)', os.path.basename(path)).group(1)) if re.search(r'(\d+)', os.path.basename(path)) else float('inf'))
    h5_files.extend(files)

#Reading the features from the h5 files and creating and index containing all features
index_files = [np.array([], dtype=object)] * len(h5_files)
for idx, file_path in enumerate(h5_files):
    with h5py.File(file_path, 'r') as f:
        index_files[idx] = f['feats'][0:feat_len]


#Setting up the matrix containg all features of all 1139 cells, having one row in the matrix contain every feature of one cell
num_samples = min(1139, len(index_files))
d = np.zeros((num_samples, feat_len))
for i in range(num_samples):
    for s in range(feat_len):
        d[i, s] = index_files[i][0, s]


#Setting up the labels and providing the annotation        
labels=["normal","Rouleaux","Echinocyte","teardrop","schistocyte","crenatedcells","ecocentric"]
annodata=["normal","normal","rouleaux","echinocyte","rouleaux","echinocyte","teardrop","teardrop","echinocyte","normal","echinocyte","schistocyte","normal","schistocyte","rouleaux","teardrop","rouleaux","rouleaux","normal","normal","normal","crenatedcells","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","schistocyte","normal","normal","echinocyte","crenatedcells","schistocyte","normal","rouleaux","schistocyte","teardrop","echinocyte","rouleaux","normal","echinocyte","normal","rouleaux","echinocyte","schistocyte","echinocyte","echinocyte","teardrop","schistocyte","echinocyte","normal","schistocyte","crenatedcells","crenatedcells","normal","teardrop","echinocyte","normal","rouleaux","crenatedcells","rouleaux","teardrop","rouleaux","rouleaux","schistocyte","teardrop","schistocyte","echinocyte","echinocyte","crenatedcells","echinocyte","normal","crenatedcells","echinocyte","schistocyte","echinocyte","rouleaux","normal","rouleaux","crenatedcells","rouleaux","teardrop","rouleaux","normal","teardrop","ecocentric","normal","normal","echinocyte","echinocyte","crenatedcells","ecocentric","crenatedcells","teardrop","schistocyte","teardrop","normal","normal","echinocyte","schistocyte","echinocyte","schistocyte","crenatedcells","teardrop","rouleaux","rouleaux","rouleaux","schistocyte","normal","teardrop","schistocyte","echinocyte","rouleaux","normal","normal","rouleaux","normal","rouleaux","rouleaux","rouleaux","normal","teardrop","rouleaux","rouleaux","normal","ecocentric","teardrop","ecocentric","ecocentric","teardrop","normal","normal","normal","ecocentric","normal","normal","normal","normal","ecocentric","normal","normal","normal","ecocentric","crenatedcells","normal","normal","normal","normal","schistocyte","normal","crenatedcells","crenatedcells","teardrop","normal","teardrop","ecocentric","normal","crenatedcells","ecocentric","normal","schistocyte","normal","normal","normal","normal","normal","normal","ecocentric","schistocyte","normal","ecocentric","crenatedcells","schistocyte","normal","normal","normal","normal","normal","normal","normal","ecocentric","normal","normal","normal","normal","normal","normal","normal","normal","normal","ecocentric","normal","normal","normal","ecocentric","normal","schistocyte","normal","normal","schistocyte","normal","normal","normal","normal","ecocentric","schistocyte","schistocyte","teardrop","normal","normal","normal","schistocyte","normal","normal","normal","normal","normal","schistocyte","schistocyte","normal","normal","teardrop","schistocyte","echinocyte","normal","normal","normal","normal","normal","normal","schistocyte","normal","normal","normal","normal","normal","normal","normal","normal","normal","normal","ecocentric","schistocyte","ecocentric","normal","normal","schistocyte","normal","normal","normal","normal","normal","normal","normal","normal","normal","normal","teardrop","normal","schistocyte","teardrop","normal","ecocentric","schistocyte","normal","schistocyte","normal","schistocyte","normal","normal","normal","normal","normal","normal","teardrop","normal","normal","normal","teardrop","crenatedcells","teardrop","schistocyte","teardrop","teardrop","schistocyte","teardrop","normal","normal","ecocentric","teardrop","normal","normal","normal","schistocyte","teardrop","ecocentric","schistocyte","schistocyte","normal","ecocentric","schistocyte","schistocyte","normal","normal","teardrop","teardrop","schistocyte","normal","crenatedcells","crenatedcells","ecocentric","teardrop","crenatedcells","ecocentric","teardrop","normal","normal","schistocyte","ecocentric","normal","teardrop","normal","schistocyte","normal","crenatedcells","crenatedcells","normal","teardrop","schistocyte","crenatedcells","schistocyte","teardrop","crenatedcells","schistocyte","teardrop","normal","teardrop","normal","schistocyte","normal","schistocyte","crenatedcells","normal","crenatedcells","ecocentric","ecocentric","teardrop","schistocyte","schistocyte","normal","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","normal","schistocyte","schistocyte","ecocentric","schistocyte","normal","normal","normal","normal","normal","schistocyte","teardrop","teardrop","ecocentric","schistocyte","normal","normal","ecocentric","schistocyte","schistocyte","schistocyte","teardrop","schistocyte","schistocyte","schistocyte","normal","schistocyte","crenatedcells","normal","ecocentric","teardrop","schistocyte","schistocyte","crenatedcells","teardrop","teardrop","ecocentric","normal","crenatedcells","schistocyte","schistocyte","teardrop","teardrop","teardrop","teardrop","teardrop","schistocyte","teardrop","normal","schistocyte","schistocyte","schistocyte","schistocyte","ecocentric","teardrop","ecocentric","teardrop","ecocentric","crenatedcells","ecocentric","schistocyte","ecocentric","teardrop","schistocyte","schistocyte","crenatedcells","schistocyte","schistocyte","schistocyte","ecocentric","ecocentric","normal","ecocentric","schistocyte","crenatedcells","ecocentric","ecocentric","teardrop","schistocyte","teardrop","schistocyte","schistocyte","normal","teardrop","teardrop","schistocyte","ecocentric","schistocyte","schistocyte","schistocyte","crenatedcells","normal","crenatedcells","schistocyte","teardrop","teardrop","schistocyte","ecocentric","ecocentric","teardrop","crenatedcells","normal","ecocentric","ecocentric","normal","normal","ecocentric","normal","schistocyte","normal","schistocyte","schistocyte","teardrop","ecocentric","ecocentric","crenatedcells","normal","ecocentric","normal","ecocentric","normal","normal","normal","teardrop","ecocentric","normal","schistocyte","teardrop","ecocentric","ecocentric","ecocentric","ecocentric","crenatedcells","normal","normal","teardrop","ecocentric","ecocentric","normal","normal","normal","schistocyte","normal","normal","teardrop","schistocyte","schistocyte","teardrop","schistocyte","teardrop","normal","crenatedcells","teardrop","teardrop","normal","teardrop","crenatedcells","teardrop","teardrop","normal","normal","teardrop","teardrop","teardrop","teardrop","schistocyte","teardrop","ecocentric","ecocentric","teardrop","teardrop","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","crenatedcells","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","echinocyte","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","ecocentric","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","rouleaux","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","schistocyte","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop","teardrop"]

#Setting up the Random Forest Classifier
encoder = LabelEncoder()
encodedanno = encoder.fit_transform(annodata)
file_names = np.array(h5_files[:num_samples])
Train1, Val1, Train2, Val2, Train_files, Val_files = train_test_split(d, encodedanno, file_names, test_size=0.2, random_state=42, stratify=encodedanno)
classifier2= RandomForestClassifier(n_estimators=100, random_state=42)
classifier2.fit(Train1, Train2)
pred2 = classifier2.predict(Val1)
decoded_preds = encoder.inverse_transform(pred2)
decoded_true = encoder.inverse_transform(Val2)

#Printing the prediction results of the validation set for increased transparancy and 
for file, pred, true in zip(Val_files, decoded_preds, decoded_true):
    print(f"{os.path.basename(file)} - Predicted: {pred}, Actual: {true}")
    
#Printing the classification report for the Random Forest
classcheck2 = np.arange(len(encoder.classes_))
print(classification_report(Val2, pred2,labels=classcheck2, target_names=encoder.classes_))

#Setting up and displaying the confusion matrix
confusionmatrix2 = confusion_matrix(Val2, pred2, labels=np.arange(len(encoder.classes_)))
sns.heatmap(confusionmatrix2, annot=True,fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("predicted")
plt.ylabel("true")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
