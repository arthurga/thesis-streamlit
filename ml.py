# Core Packages
import streamlit as st
import streamlit.components.v1 as components
import json
import requests
from sodapy import Socrata
from collections import Counter

# Utils
import pandas as pd
import numpy as np
import joblib
import os
from eda import filtered_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import pickle

# Geocoding Address
from shapely.geometry import Point, Polygon
import geopandas as gpd
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Classifiers
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

# from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from scipy import special


#Precision
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report, confusion_matrix

from home_page import filtered_data


df = filtered_data()
df = df.drop('ucr_group', axis=1)


gender_dict = {'FEMALE': 0, 'F - FEMALE': 0, 'MALE': 1, 'M - MALE': 1, 'UNKNOWN': 2, 'NON-PERSON (BUSINESS': 2}

age_dict = {'UNDER 18': 0, 'JUVENILE (UNDER 18)': 0, '18-25': 1, '26-30': 2, '31-40': 3, '41-50': 4, '51-60': 5, '61-70': 6, 'OVER 70': 7, 'UNKNOWN': 8, 'ADULT (18+)': 8, '00': 8}

race_dict = {'WHITE': 0, 'BLACK': 1, 'ASIAN/PACIFIC ISLAND': 2, 'AMERICAN INDIAN/ALAS': 3, 'UNKNOWN': 4,
             'ASIAN OR PACIFIC ISL': 5, 'AMERICAN IINDIAN/ALA': 6, 'HISPANIC': 7}

target_dict = {'D--VICTIM REFUSED TO COOPERATE': 0, 'Z--EARLY CLOSED': 0,'J--CLOSED': 0, 'H--WARRANT ISSUED': 0,
                 'F--CLEARED BY ARREST - ADULT': 1, 'K--UNFOUNDED': 0, 'G--CLEARED BY ARREST - JUVENILE': 1,
                 'I--INVESTIGATION PENDING': 0, 'B--PROSECUTION DECLINED': 0, 'E--JUVENILE/NO CUSTODY': 0,
                 'A--DEATH OF OFFENDER': 0, 'U--UNKNOWN': 0, 'C--EXTRADITION DENIED': 0}

week_dict = {'SUNDAY': 6, 'MONDAY': 0, 'TUESDAY': 1, 'WEDNESDAY': 2, 'THURSDAY': 3, 'FRIDAY': 4, 'SATURDAY': 5}


def age_encode(val):
    return age_dict[val]

def gender_encode(val):
    return gender_dict[val]

def race_encode(val):
    return race_dict[val]

def target_encode(val):
    return target_dict[val]

def week_encode(val):
    return week_dict[val]

df['victim_age'] = df['victim_age'].apply(age_encode)
df['victim_gender'] = df['victim_gender'].apply(gender_encode)
df['victim_race'] = df['victim_race'].apply(race_encode)
df['dayofweek'] = df['dayofweek'].apply(week_encode)
df['clsd'] = df['clsd'].apply(target_encode)


def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value


# Basic preprocessing required for all the models.
def preprocessing(df):
    # Assign X and y
    X = df.drop('clsd', axis=1)
    y = df['clsd']

    smt = SMOTE(random_state=42)
    X_train_res, y_train_res = smt.fit_resample(X, y)

    # 1. Splitting X,y into Train & Test
    X_train, X_test, y_train, y_test = train_test_split(X_train_res, y_train_res, test_size=0.25, random_state=42)

    from sklearn.preprocessing import StandardScaler
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_test = feature_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# Training Decission Tree for Classification
@st.cache(suppress_st_warning=True)
def decisionTree(X_train, X_test, y_train, y_test):
    # Train the model
    tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    r_score = metrics.recall_score(y_test, y_pred)
    p_score = metrics.precision_score(y_test, y_pred)
    f_score = metrics.f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred) 

    return score,  p_score, r_score, f_score, report


def logisticRegression(X_train, X_test, y_train, y_test):
    # Train the model
    lr = LogisticRegression(C = 0.1, solver = 'liblinear')
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    r_score = metrics.recall_score(y_test, y_pred)
    p_score = metrics.precision_score(y_test, y_pred)
    f_score = metrics.f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return score, p_score, r_score, f_score, report


def XGBRegressor(X_train, X_test, y_train, y_test):
    # Train the model
    xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
    xg_reg.fit(X_train, y_train)
    y_pred = xg_reg.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    r_score = metrics.recall_score(y_test, y_pred)
    p_score = metrics.precision_score(y_test, y_pred)
    f_score = metrics.f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return score, p_score, r_score, f_score, report


@st.cache(suppress_st_warning=True)
def SVC(X_train, X_test, y_train, y_test):
    # Train the model
    svc_clf = LinearSVC()
    svc_clf.fit(X_train, y_train)
    y_pred = svc_clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    r_score = metrics.recall_score(y_test, y_pred)
    p_score = metrics.precision_score(y_test, y_pred)
    f_score = metrics.f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return score, p_score, r_score, f_score, report


# Training KNN Classifier
@st.cache(suppress_st_warning=True)
def Knn_Classifier(X_train, X_test, y_train, y_test):
    k_clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_neighbors=5, p=2,weights='uniform')
    k_clf.fit(X_train, y_train)
    y_pred = k_clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    r_score = metrics.recall_score(y_test, y_pred)
    p_score = metrics.precision_score(y_test, y_pred)
    f_score = metrics.f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return score, p_score, r_score, f_score, report, k_clf


# Training Neural Network for Classification.
@st.cache(suppress_st_warning=True)
def neuralNet(X_train, X_test, y_train, y_test):
    # Instantiate the Classifier and fit the model.
    n_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    n_clf.fit(X_train, y_train)
    y_pred = n_clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    r_score = metrics.recall_score(y_test, y_pred)
    p_score = metrics.precision_score(y_test, y_pred)
    f_score = metrics.f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return score, p_score, r_score, f_score, report


# Training Gradient Boosting Classifier
@st.cache(suppress_st_warning=True)
def GradientBoosting(X_train, X_test, y_train, y_test):
    g_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    g_clf.fit(X_train, y_train)
    y_pred = g_clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    r_score = metrics.recall_score(y_test, y_pred)
    p_score = metrics.precision_score(y_test, y_pred)
    f_score = metrics.f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return score, p_score, r_score, f_score, report



# Evaluate Model Accuracy

# Model
models = []
model_names = []
model_score = []
model_precision = []
model_recall = []
model_f1 = []
model_report = []
all_models = []
scoring = "Accuracy"


def user_input_features():
    # Displays the user input features
    st.sidebar.subheader('User Input features')
    gender = st.sidebar.radio('Gender', ('MALE', 'FEMALE'))
    race = st.sidebar.selectbox('Race', ('WHITE', 'BLACK', 'ASIAN/PACIFIC ISLAND', 'AMERICAN INDIAN/ALAS',
                                         'UNKNOWN', 'ASIAN OR PACIFIC ISL', 'AMERICAN IINDIAN/ALA', 'HISPANIC'))
    age = st.sidebar.selectbox('Age', ('UNDER 18', '18-25', '26-30', '31-40', '41-50', '51-60', '61-70', 'OVER 70'))
    street = st.sidebar.text_input("Street", "611 Foulke St")
    city = st.sidebar.text_input("City", "Cincinnati")
    state = st.sidebar.text_input("State", "Ohio")
    country = st.sidebar.text_input("Country", "USA")

    geolocator = Nominatim(user_agent="GTA Lookup")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geolocator.geocode(street + ", " + city + ", " + state + ", " + country)

    lat = location.latitude
    lon = location.longitude

    data = {'address':street + ", " + city + ", " + state + ", " + country,
            'gender': gender,
            'race': race,
            'age': age,
            'longitude': lon,
            'latitude': lat}
    features = ([data])
    return features


def run_ml():
    st.title('Model Prediction')
    user_df = user_input_features()
    data = user_df[0]

    #Encoding user selection
    age_en = get_value(data['age'], age_dict)
    gender_en = get_value(data['gender'], gender_dict)
    race_en = get_value(data['race'], race_dict)
    lat_en = data['latitude']
    lon_en = data['longitude']
    now = pd.to_datetime('now')
    year_en = now.year
    month_en = now.month
    day_en = now.day
    hour_en = now.hour
    minute_en = now.minute
    week_en = now.weekday()


    #storing the inputs
    selected_sample = [year_en, month_en, day_en, hour_en, minute_en, week_en, age_en, race_en, gender_en, lat_en, lon_en]
    sample = np.array(selected_sample).reshape(1, -1)
    st.subheader('User data')
    st.write(sample)

    #Process Data
    X_train, X_test, y_train, y_test = preprocessing(df)


    models.append(("K-Nearest Neighbors", Knn_Classifier(X_train, X_test, y_train, y_test)))
    models.append(("Support Vector Machine", SVC(X_train, X_test, y_train, y_test)))
    models.append(("Decision Tree", decisionTree(X_train, X_test, y_train, y_test)))
    models.append(("Neural Net", neuralNet(X_train, X_test, y_train, y_test)))
    models.append(("Gradient Boosting", neuralNet(X_train, X_test, y_train, y_test)))

    for name, model in models:
        score = model
        model_names.append(name)
        model_score.append(score[0])
        model_precision.append(score[1])
        model_recall.append(score[2])
        model_f1.append(score[3])
        accuracy_results = {"model_name": name, "model_accuracy": score[0], "model_precision": score[1],
                            "model_recall": score[2], "model_f1": score[3]}
        all_models.append(accuracy_results)

        metDf = pd.DataFrame(zip(model_names, model_score, model_precision, model_recall, model_f1))
        metDf.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 score"]


    with st.beta_expander("Model Performance"):
        st.dataframe(metDf)

    # ML Section
    choose_model = st.sidebar.selectbox("Choose the ML Model",
                                        ["ALL", "Decision Tree", "K-Nearest Neighbours", "SVC Classifier",
                                         "Neural Net", "Gradient Boosting", "Logistic Regression"])
    if (choose_model == "Decision Tree"):
        score, r_score, p_score, f_score, report = decisionTree(X_train, X_test, y_train, y_test)
        st.text("Decision Tree results: ")
        dtDf = pd.DataFrame(columns=["Accuracy", "Recall", "Precision", "F1 score"])
        dtDf.loc[0] = [score, r_score, p_score, f_score]
        st.dataframe(dtDf)


    elif (choose_model == "Logistic Regression"):
        score, r_score, p_score, f_score, report = logisticRegression(X_train, X_test, y_train, y_test)
        st.text("Logistic Regression results: ")
        lDf = pd.DataFrame(columns=["Accuracy", "Recall", "Precision", "F1 score"])
        lDf.loc[0] = [score, r_score, p_score, f_score]
        st.dataframe(lDf)


    elif (choose_model == "XGBRegressor"):
        score, r_score, p_score, f_score, report = XGBRegressor(X_train, X_test, y_train, y_test)
        st.text("XGBRegressor results: ")
        xgbDf = pd.DataFrame(columns=["Accuracy", "Recall", "Precision", "F1 score"])
        xgbDf.loc[0] = [score, r_score, p_score, f_score]
        st.dataframe(xgbDf)


    elif (choose_model == "K-Nearest Neighbours"):
        score, r_score, p_score, f_score, report, k_clf = Knn_Classifier(X_train, X_test, y_train, y_test)
        st.text("KNN results: ")
        knnDf = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1 score"])
        knnDf.loc[0] = [score, p_score, r_score, f_score]
        st.dataframe(knnDf)
        st.text("Report of KNN model is: ")
        st.write(report)
        pred = k_clf.predict(sample)
        st.write(pred)

        if pred == 1:
            st.success('The location is safe')
        else:
            st.warning('The location is not safe')


    elif (choose_model == "SVC Classifier"):
        score, report, svc_clf = SVC(X_train, X_test, y_train, y_test)
        st.text("Accuracy of SVC model is: ")
        st.write(score, "%")
        st.text("Report of SVC model is: ")
        st.write(report)


    elif (choose_model == "Neural Net"):
        score, report, clf = neuralNet(X_train, X_test, y_train, y_test)
        st.text("Accuracy of Neural Net model is: ")
        st.write(score, "%")
        st.text("Report of Neural Net model is: ")
        st.write(report)


    elif (choose_model == "Gradient Boosting"):
        score, report, g_clf = GradientBoosting(X_train, X_test, y_train, y_test)
        st.text("Accuracy of Gradient Boosting Net model is: ")
        st.write(score, "%")
        st.text("Report of Gradient Boosting Net model is: ")
        st.write(report)


    with st.beta_expander("Want to predict your own data?"):
        st.write(data)
        if st.button("Is it Safe?"):
            st.write('Under construction')


