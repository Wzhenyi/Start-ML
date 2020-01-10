from sklearn.datasets import load_digits
import numpy as np 
import matplotlib.pyplot as plt 
digits = load_digits()
# To show there are 1797 images and dimention 64=8*8
print("Image", digits.data.shape)
# To show 1797 labels
print("Label", digits.target.shape)
# To split dataset into "train" and "test"
from sklearn.model_selection import train_test_split
x_tr, x_te, y_tr, y_te = train_test_split(digits.data, digits.target, test_size=0.165, random_state=0)
# Use the model of logistic regression in sklearn
from sklearn.linear_model import LogisticRegression
LR= LogisticRegression()
LR.fit(x_tr, y_tr)
# We want to know what is the accuracy
s = LR.score(x_te, y_te)
print(s)