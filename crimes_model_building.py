import streamlit as st
import pandas as pd
import numpy as np
from sklearn import metrics
from eda import filtered_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


@st.cache
def loadData():
    df = filtered_data()
    features = ['victim_age', 'victim_race', 'victim_gender', 'lon', 'lat', 'clsd']
    crimes_df = df[features]

    return crimes_df

df = loadData()
df = df.dropna()


gender_mapper = {'FEMALE': 0, 'F - FEMALE': 0, 'MALE': 1, 'M - MALE': 1, 'UNKNOWN': 2, 'NON-PERSON (BUSINESS': 2}

age_mapper = {'UNDER 18': 0, '18-25': 1, '26-30': 2, '31-40': 3, '41-50': 4, '51-60': 5, '61-70': 6, 'OVER 70': 7, 'UNKNOWN': 8}

race_mapper = {'WHITE': 0, 'BLACK': 1, 'ASIAN/PACIFIC ISLAND': 2, 'AMERICAN INDIAN/ALAS': 3, 'UNKNOWN': 4,
             'ASIAN OR PACIFIC ISL': 5, 'AMERICAN IINDIAN/ALA': 6, 'HISPANIC': 7}

target_mapper = {'D--VICTIM REFUSED TO COOPERATE': 0, 'Z--EARLY CLOSED': 0,'J--CLOSED': 0, 'H--WARRANT ISSUED': 0,
                 'F--CLEARED BY ARREST - ADULT': 1, 'K--UNFOUNDED': 0, 'G--CLEARED BY ARREST - JUVENILE': 1,
                 'I--INVESTIGATION PENDING': 0, 'B--PROSECUTION DECLINED': 0, 'E--JUVENILE/NO CUSTODY': 0,
                 'A--DEATH OF OFFENDER': 0, 'U--UNKNOWN': 0, 'C--EXTRADITION DENIED': 0}


def age_encode(val):
    return age_mapper[val]

def gender_encode(val):
    return gender_mapper[val]

def race_encode(val):
    return race_mapper[val]

def target_encode(val):
    return target_mapper[val]

df['victim_age'] = df['victim_age'].apply(age_encode)
df['victim_gender'] = df['victim_gender'].apply(gender_encode)
df['victim_race'] = df['victim_race'].apply(race_encode)
df['clsd'] = df['clsd'].apply(target_encode)


# Separating X and y
X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values


# 1. Splitting X,y into Train & Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# Model
models = []
models.append(("Logistic Regression", LogisticRegression()))
models.append(("Linear Discriminant Analysis", LinearDiscriminantAnalysis()))
models.append(("K-Nearest Neighbors", KNeighborsClassifier()))
models.append(("Classification And Regression Tree", DecisionTreeClassifier()))
models.append(("Naive Bayes", GaussianNB()))
models.append(("Support Vector Machine", SVC()))
models.append(("SMOTE", SMOTE()))


# Evaluate Model Accuracy

# List
model_names = []
model_mean = []
model_std = []
all_models = []
scoring = "Accuracy"



for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=None)
    cv_results = model_selection.cross_val_score(model,X,y,cv=kfold)
    model_names.append(name)
    model_mean.append(cv_results.mean())
    model_std.append(cv_results.std())

    accuracy_results = {"model_name": name,"model_accuracy":cv_results.mean(), "standard_deviation":cv_results.std()}
    all_models.append(accuracy_results)


    metDf = pd.DataFrame(zip(model_names,model_mean,model_std))
    metDf.columns = ["Model","Accuracy","Standard Deviation"]
    output = st.dataframe(metDf)




# Saving the model
import pickle

# Save to file in the current working directory
pkl_filename = "CincyCrimes_clf.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(output, file)

