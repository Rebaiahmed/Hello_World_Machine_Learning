import pandas as pd
from pandas import Series,DataFrame
#numpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


#machine learning import our classifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import  RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# get titanic & test csv files as a DataFrame
titanic_train = pd.read_csv("./data/train.csv")
test_df    = pd.read_csv("./data/test.csv")


print(titanic_train.sample(3))

#visualize the data
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=titanic_train);



sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=titanic_train,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"]);
