# Core Packages
import streamlit as st
import streamlit.components.v1 as components
import json
import requests
import seaborn as sns
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
import matplotlib
import matplotlib.pyplot as plt

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from scipy import special


#Precision
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, log_loss
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.model_selection import cross_val_score

from home_page import filtered_data


@st.cache(suppress_st_warning=True)
def ml_data():
	df = filtered_data()
	return df

df = ml_data()
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

def get_key(val,my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

# Basic preprocessing required for all the models.
def preprocessing(df):
    # Assign X and y
    X = df.drop('clsd', axis=1)
    y = df['clsd']

    smt = SMOTE()
    X_train_res, y_train_res = smt.fit_resample(X, y)

    tomeklinks = TomekLinks()
    X_tl, y_tl = tomeklinks.fit_resample(X, y)


    # 1. Splitting X,y into Train & Test
    X_train, X_test, y_train, y_test = train_test_split(X_train_res, y_train_res, test_size=0.20, random_state=7)

    return X_train, X_test, y_train, y_test



# Training Decission Tree for Classification
@st.cache(suppress_st_warning=True)
def DecisionTree(X_train, X_test, y_train, y_test):
    # Train the model
    tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
    tree.fit(X_train, y_train)

    #Train
    y_pred_train = tree.predict(X_train)
    train_score = metrics.accuracy_score(y_train, y_pred_train) * 100
    train_r_score = metrics.recall_score(y_train, y_pred_train)
    train_p_score = metrics.precision_score(y_train, y_pred_train)
    train_f_score = metrics.f1_score(y_train, y_pred_train)
    train_report = classification_report(y_train, y_pred_train)
    train_logloss = log_loss(y_train, y_pred_train)
    train_cm = confusion_matrix(y_train, y_pred_train)

    #Test
    y_pred_test = tree.predict(X_test)
    test_score = metrics.accuracy_score(y_test, y_pred_test) * 100
    test_r_score = metrics.recall_score(y_test, y_pred_test)
    test_p_score = metrics.precision_score(y_test, y_pred_test)
    test_f_score = metrics.f1_score(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test)
    test_logloss = log_loss(y_train, y_pred_train)
    test_cm = confusion_matrix(y_test, y_pred_test)

    return train_score, train_p_score, train_r_score, train_f_score, train_logloss, train_cm, train_report, \
           test_score, test_p_score, test_r_score, test_f_score, test_logloss, test_cm, test_report, tree


@st.cache(suppress_st_warning=True)
def SupportVector(X_train, X_test, y_train, y_test):
    # Train the model
    svm_clf = LinearSVC()
    svm_clf.fit(X_train, y_train)

    #Train
    y_pred_train = svm_clf.predict(X_train)
    train_score = metrics.accuracy_score(y_train, y_pred_train) * 100
    train_r_score = metrics.recall_score(y_train, y_pred_train)
    train_p_score = metrics.precision_score(y_train, y_pred_train)
    train_f_score = metrics.f1_score(y_train, y_pred_train)
    train_report = classification_report(y_train, y_pred_train)
    train_logloss = log_loss(y_train, y_pred_train)
    train_cm = confusion_matrix(y_train, y_pred_train)

    #Test
    y_pred_test = svm_clf.predict(X_test)
    test_score = metrics.accuracy_score(y_test, y_pred_test) * 100
    test_r_score = metrics.recall_score(y_test, y_pred_test)
    test_p_score = metrics.precision_score(y_test, y_pred_test)
    test_f_score = metrics.f1_score(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test)
    test_logloss = log_loss(y_train, y_pred_train)
    test_cm = confusion_matrix(y_test, y_pred_test)

    return train_score, train_p_score, train_r_score, train_f_score, train_logloss, train_cm, train_report, \
           test_score, test_p_score, test_r_score, test_f_score, test_logloss, test_cm, test_report, svm_clf


@st.cache(allow_output_mutation=True)
def RandomForest(X_train, X_test, y_train, y_test):
    # Train the model
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)

    #Train
    y_pred_train = rf_clf.predict(X_train)
    train_score = metrics.accuracy_score(y_train, y_pred_train) * 100
    train_r_score = metrics.recall_score(y_train, y_pred_train)
    train_p_score = metrics.precision_score(y_train, y_pred_train)
    train_f_score = metrics.f1_score(y_train, y_pred_train)
    train_report = classification_report(y_train, y_pred_train)
    train_logloss = log_loss(y_train, y_pred_train)
    train_cm = confusion_matrix(y_train, y_pred_train)

    #Test
    y_pred_test = rf_clf.predict(X_test)
    test_score = metrics.accuracy_score(y_test, y_pred_test) * 100
    test_r_score = metrics.recall_score(y_test, y_pred_test)
    test_p_score = metrics.precision_score(y_test, y_pred_test)
    test_f_score = metrics.f1_score(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test)
    test_logloss = log_loss(y_train, y_pred_train)
    test_cm = confusion_matrix(y_test, y_pred_test)

    return train_score, train_p_score, train_r_score, train_f_score, train_logloss, train_cm, train_report, \
           test_score, test_p_score, test_r_score, test_f_score, test_logloss, test_cm, test_report, rf_clf


@st.cache(suppress_st_warning=True)
def logisticRegression(X_train, X_test, y_train, y_test):
    # Train the model
    lr = LogisticRegression(C = 0.1, solver = 'liblinear')
    lr.fit(X_train, y_train)
    y_pred_train = lr.predict(X_train)
    train_score = metrics.accuracy_score(y_train, y_pred_train) * 100
    train_r_score = metrics.recall_score(y_train, y_pred_train)
    train_p_score = metrics.precision_score(y_train, y_pred_train)
    train_f_score = metrics.f1_score(y_train, y_pred_train)
    train_report = classification_report(y_train, y_pred_train)
    train_logloss = log_loss(y_train, y_pred_train)
    train_cm = confusion_matrix(y_train, y_pred_train)

    # Test
    y_pred_test = lr.predict(X_test)
    test_score = metrics.accuracy_score(y_test, y_pred_test) * 100
    test_r_score = metrics.recall_score(y_test, y_pred_test)
    test_p_score = metrics.precision_score(y_test, y_pred_test)
    test_f_score = metrics.f1_score(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test)
    test_logloss = log_loss(y_train, y_pred_train)
    test_cm = confusion_matrix(y_test, y_pred_test)

    return train_score, train_p_score, train_r_score, train_f_score, train_logloss, train_cm, train_report, \
           test_score, test_p_score, test_r_score, test_f_score, test_logloss, test_cm, test_report, lr



# Training KNN Classifier
@st.cache(suppress_st_warning=True)
def KNN_Classifier(X_train, X_test, y_train, y_test):
    k_clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_neighbors=5, p=2,weights='uniform')
    k_clf.fit(X_train, y_train)
    y_pred_train = k_clf.predict(X_train)
    train_score = metrics.accuracy_score(y_train, y_pred_train) * 100
    train_r_score = metrics.recall_score(y_train, y_pred_train)
    train_p_score = metrics.precision_score(y_train, y_pred_train)
    train_f_score = metrics.f1_score(y_train, y_pred_train)
    train_report = classification_report(y_train, y_pred_train)
    train_logloss = log_loss(y_train, y_pred_train)
    train_cm = confusion_matrix(y_train, y_pred_train)

    # Test
    y_pred_test = k_clf.predict(X_test)
    test_score = metrics.accuracy_score(y_test, y_pred_test) * 100
    test_r_score = metrics.recall_score(y_test, y_pred_test)
    test_p_score = metrics.precision_score(y_test, y_pred_test)
    test_f_score = metrics.f1_score(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test)
    test_logloss = log_loss(y_train, y_pred_train)
    test_cm = confusion_matrix(y_test, y_pred_test)

    return train_score, train_p_score, train_r_score, train_f_score, train_logloss, train_cm, train_report, \
           test_score, test_p_score, test_r_score, test_f_score, test_logloss, test_cm, test_report, k_clf



@st.cache(suppress_st_warning=True)
def NaiveBayes(X_train, X_test, y_train, y_test):
    # Train the model
    gnb_clf = GaussianNB()
    gnb_clf.fit(X_train, y_train)
    y_pred_train = gnb_clf.predict(X_train)
    train_score = metrics.accuracy_score(y_train, y_pred_train) * 100
    train_r_score = metrics.recall_score(y_train, y_pred_train)
    train_p_score = metrics.precision_score(y_train, y_pred_train)
    train_f_score = metrics.f1_score(y_train, y_pred_train)
    train_report = classification_report(y_train, y_pred_train)
    train_logloss = log_loss(y_train, y_pred_train)
    train_cm = confusion_matrix(y_train, y_pred_train)

    # Test
    y_pred_test = gnb_clf.predict(X_test)
    test_score = metrics.accuracy_score(y_test, y_pred_test) * 100
    test_r_score = metrics.recall_score(y_test, y_pred_test)
    test_p_score = metrics.precision_score(y_test, y_pred_test)
    test_f_score = metrics.f1_score(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test)
    test_logloss = log_loss(y_train, y_pred_train)
    test_cm = confusion_matrix(y_test, y_pred_test)

    return train_score, train_p_score, train_r_score, train_f_score, train_logloss, train_cm, train_report, \
           test_score, test_p_score, test_r_score, test_f_score, test_logloss, test_cm, test_report, gnb_clf


# Evaluate Model Accuracy

# Model
models = []
model_names = []
train_model_score = []
train_model_precision = []
train_model_recall = []
train_model_f1 = []
train_model_report = []
train_model_logloss = []
train_model_cm = []
all_train_models = []

test_model_score = []
test_model_precision = []
test_model_recall = []
test_model_f1 = []
test_model_report = []
test_model_logloss = []
test_model_cm = []
all_test_models = []

scoring = "Accuracy"


def user_input_features():
    # Displays the user input features
    st.write("Enter location address: ")
    street = st.text_input("Street", "611 Foulke St")

    c1, c2 = st.beta_columns(2)
    with c1:
        city = st.text_input("City", "Cincinnati")
        gender = st.radio('Gender', ('MALE', 'FEMALE'))
        race = st.selectbox('Race', ('WHITE', 'BLACK', 'ASIAN/PACIFIC ISLAND', 'AMERICAN INDIAN/ALAS',
                                             'UNKNOWN', 'ASIAN OR PACIFIC ISL', 'AMERICAN IINDIAN/ALA', 'HISPANIC'))
    with c2:
        state = st.text_input("State", " Ohio")
        age = st.selectbox('Age', ('UNDER 18', '18-25', '26-30', '31-40', '41-50', '51-60', '61-70', 'OVER 70'))
        week = st.selectbox('Day of Week', ('SUNDAY', 'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY'))

    geolocator = Nominatim(user_agent="GTA Lookup")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geolocator.geocode(street + ", " + city + ", " + state + ", USA")

    lat = location.latitude
    lon = location.longitude

    today = datetime.today()

    #
    st.sidebar.subheader('Select datetime')
    year = st.sidebar.slider('year', 2010, 2025, today.year)
    month = st.sidebar.slider('month', 1, 12, today.month)
    day = st.sidebar.slider('day', 1, 31, today.day)
    hour = st.sidebar.slider('hour', 1, 24, today.hour)

    data = {'address': street + ", " + city + "," + state + ", USA",
            'gender': gender,
            'race': race,
            'age': age,
            'week': week,
            'year': year,
            'month': month,
            'day': day,
            'hour': hour,
            'longitude': lon,
            'latitude': lat}
    features = ([data])
    return features


def run_ml():
    st.title('Model Performance')
    submenu = st.sidebar.selectbox("Submenu", ["Model Performance", "User Prediction"])
    #Process Data
    X = df.drop('clsd', axis=1)
    y = df['clsd']
    X_train, X_test, y_train, y_test = preprocessing(df)
    class_names = ['No Arrest', 'Arrest']

    models.append(("K-Nearest Neighbors", KNN_Classifier(X_train, X_test, y_train, y_test)))
    models.append(("Decision Tree", DecisionTree(X_train, X_test, y_train, y_test)))
    models.append(("Support Vector", SupportVector(X_train, X_test, y_train, y_test)))
    models.append(("Logistic Regression", logisticRegression(X_train, X_test, y_train, y_test)))
    models.append(("Naive Bayes", NaiveBayes(X_train, X_test, y_train, y_test)))
    models.append(("Random Forest", RandomForest(X_train, X_test, y_train, y_test)))

    for name, model in models:
        score = model
        model_names.append(name)
        train_model_score.append(score[0])
        train_model_precision.append(score[1])
        train_model_recall.append(score[2])
        train_model_f1.append(score[3])
        train_model_logloss.append(score[4])
        train_model_cm.append(score[5])
        train_model_report.append(score[6])
        train_accuracy_results = {"model_name": name, "model_accuracy": score[0], "model_precision": score[1], "model_recall": score[2],
                                  "model_f1": score[3], "model_logloss": score[4],"model_report":score[6], "model_param": score[14]}
        all_train_models.append(train_accuracy_results)

        test_model_score.append(score[7])
        test_model_precision.append(score[8])
        test_model_recall.append(score[9])
        test_model_f1.append(score[10])
        test_model_logloss.append(score[11])
        test_model_cm.append(score[12])
        test_accuracy_results = {"model_name": name, "model_accuracy": score[7], "model_precision": score[8], "model_recall": score[9],
                                 "model_f1": score[10], "model_logloss": score[11],"model_report": score[13], "model_param": score[14]}
        all_test_models.append(test_accuracy_results)
        pass


    metDf_train = pd.DataFrame(all_train_models).sort_values(by=['model_accuracy'], ascending=False)
    metDf_test = pd.DataFrame(all_test_models).sort_values(by=['model_accuracy'], ascending=False)
    metDf_train.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 score", "Log Loss", "Model Report", "Model Parameter"]
    metDf_test.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 score", "Log Loss", "Model Report", "Model Parameter"]


    top_param = metDf_test['Model Parameter'].iloc[0]
    model_report = metDf_test['Model Report'].iloc[0]


    if submenu == 'Model Performance':
        st.markdown('**1.2. Data Splits**')
        st.write('Training set:', X_train.shape)
        st.write('Test set:', y_train.shape)
        st.markdown('**1.3. Variable Details**')
        st.write('X variables')
        st.info(list(X_train.columns))
        st.write('y variable')
        st.info(y_train.name)

        # page = st.radio(
        #     'Select re-sampling data',
        #     ['Over-sampling', 'Under-sampling'],
        #     index = 0
        # )
        # if page == 'Over-sampling':
        #train
        st.markdown('**1.4. Training Set**')
        st.write('Training')
        st.dataframe(metDf_train.iloc[:, :-2])
        #test
        st.markdown('**1.5. Test Set**')
        st.write('Testing')
        st.dataframe(metDf_test.iloc[:, :-2])
        st.write("--------------------------------------------------------------------------------\n")

        st.markdown(f"**1.6. Best Model: {metDf_test.Model.iloc[0]} **")



        rf_cv = RandomForestClassifier()
        cv_scores = cross_val_score(rf_cv, X, y, cv=10)
        st.write(f'K-Fold = 10')
        st.write(f"cv_score:", cv_scores.mean())
        st.write(model_report)


        st.write("--------------------------------------------------------------------------------\n")
        st.write('**1.7. Model Graphs**')
        c1, c2 = st.beta_columns(2)
        with c1:
            st.write('Confusion Matrix')
            fig, ax = plt.subplots()
            plot_confusion_matrix(top_param, X_test, y_test, ax=ax)
            st.write(fig)

        with c2:
            st.write('ROC Curve')
            fig2, ax2 = plt.subplots()
            plot_roc_curve(top_param, X_test, y_test, ax=ax2)
            st.write(fig2)





    elif submenu == 'User Prediction':
        user_df = user_input_features()
        data = user_df[0]
        st.subheader('User input data: ')

        st.write(data)

        # Encoding user selection
        age_en = get_value(data['age'], age_dict)
        gender_en = get_value(data['gender'], gender_dict)
        race_en = get_value(data['race'], race_dict)
        lat_en = data['latitude']
        lon_en = data['longitude']
        year_en = data['year']
        month_en = data['month']
        day_en = data['day']
        hour_en = data['hour']
        week_en = get_value(data['week'], week_dict)

        # storing the inputs
        selected_sample = [[year_en, month_en, day_en, hour_en, week_en, age_en, race_en, gender_en, lat_en,
                           lon_en]]
        sample = np.array(selected_sample).reshape(1, -1)
        st.subheader('Encoded data')
        st.write(sample)

        prediction = top_param.predict_proba(selected_sample)
        safety_score = prediction[0][0]


        st.subheader('Prediction')
        if safety_score > .746:
            st.success(f"Your safety score is {safety_score*100}%")
        else:
            st.warning(f"Your safety score is {safety_score*100}%")

