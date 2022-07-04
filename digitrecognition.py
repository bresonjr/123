from cgi import test
from inspect import Attribute
from click import option
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from PIL import Image 
import PIL.ImageOps
import os,ssl,time

X,y=fetch_openml('mnist_784',version=1,return_X_y=True)
print(pd.Series(y).value_counts())
classes=['0','1','2','3','4','5','6','7','8','9']
nclasses=len(classes)

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and 
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=7500,test_size=7500)
X_train_selection=X_train/0.255
X_test_selection=X_test/0.255

clf=LogisticRegression(solver='saga',multi_class='multinominal').fit(X_train_selection,y_train)

y_pred=accuracy_score(X_test_selection)
accuracy=accuracy_score(y_train,y_pred)
print(accuracy)