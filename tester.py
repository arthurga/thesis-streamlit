# Core Packages
import streamlit as st
import streamlit.components.v1 as components
import json
import requests
from sodapy import Socrata

# EDA Packages
import pandas as pd
import numpy as np
import base64
import time
from datetime import datetime, timedelta

# Data Viz Packages
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from wordcloud import WordCloud

# Load data
@st.cache(suppress_st_warning=True)
def load_data():
    client = Socrata("data.cincinnati-oh.gov", None)
    crimes = client.get("k59e-2pvf", limit=500000)
    crimes_na = pd.DataFrame.from_records(crimes)
    crimes_df = crimes_na.replace(r'^\s*$', np.nan, regex=True)

    # convert dates to datetime
    crimes_df.date_reported = pd.to_datetime(crimes_df.date_reported, errors='coerce').dt.tz_localize('US/Eastern',ambiguous='NaT',nonexistent='NaT')
    crimes_df.date_from = pd.to_datetime(crimes_df.date_from, errors='coerce').dt.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='NaT')
    crimes_df.date_to = pd.to_datetime(crimes_df.date_to, errors='coerce').dt.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='NaT')
    crimes_df.date_of_clearance = pd.to_datetime(crimes_df.date_of_clearance, errors='coerce').dt.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='NaT')

    crimes = crimes_df.dropna(subset=['longitude_x', 'latitude_x'])
    crimes = crimes.rename(columns={'longitude_x': 'lon', 'latitude_x': 'lat'})

    crimes['lon'] = crimes['lon'].astype(float)
    crimes['lat'] = crimes['lat'].astype(float)

    crimestats = crimes

    return crimestats

# Clean data
@st.cache(suppress_st_warning=True)
def processed_data():
    data_df = load_data()
    property_crimes = ['BURGLARY/BREAKING ENTERING', 'THEFT', 'UNAUTHORIZED USE']
    crime_years = list(range(2010, 2021))
    crimes = data_df[(data_df.date_reported.dt.year.isin(crime_years)) & (data_df.ucr_group.isin(property_crimes))]
    crimes = crimes.dropna(subset=['lon','lat'])
    crimes = crimes.dropna(subset=['ucr_group'])
    crimes.drop_duplicates()

    # Breaking down date_reported into individual features
    year = crimes.date_reported.dt.year
    month = crimes.date_reported.dt.month
    hour = crimes.date_reported.dt.hour
    minutes = crimes.date_reported.dt.minute

    time_df = pd.DataFrame({'year': year,
                            'month': month,
                            'hour': hour,
                            'minute': minutes})

    final_df = pd.concat([time_df, crimes], axis=0)
    final_df = final_df.drop('date_reported', axis=1)

    return final_df



def run_home_page():
    st.title('Cincincinnati Residential Safety App')
    original_df = load_data()
    filtered_df = processed_data()

    bgcolor = st.beta_color_picker("")

    with st.beta_expander("Property Crimes", expanded=True):
        day_text = " ".join(filtered_df['ucr_group'].tolist())
        mywordcloud = WordCloud(background_color=bgcolor).generate(day_text)
        fig = plt.figure()
        plt.imshow(mywordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(fig)

    st.markdown("""
            This app analyzes the safety of city of Cincinnati neighborhoods based historical records of property crimes within the Cincinnati Metro area.
            The app builds a model by collecting user data such as age, gender, race and location coordinates to predict how safe a neighborhood is for a new resident or pedestrian, based on reported crime over the past 10 years.
            * **Python libraries:** base64, pandas, streamlit
            * **Data source:** [data.cincinnati-oh.gov](https://data.cincinnati-oh.gov/Safety/PDI-Police-Data-Initiative-Crime-Incidents/k59e-2pvf).
            * **Crime Offense:** THEFT, BURGLARY/BREAKING ENTERING, UNAUTHORIZED USE
            """)

    st.write("Show first 5 of original dataset")
    st.dataframe(original_df.head())
    st.write('Original data dimensions: ' + str(original_df.shape[0]) + ' rows and ' + str(original_df.shape[1]) + ' columns')


    st.write("Filtered data")
    now = pd.to_datetime("now")
    st.dataframe(filtered_df.head())
    st.write(now)

run_home_page()

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


  # # Breaking down date_reported into individual features
    # year = crimes_df.date_reported.dt.year
    # month = crimes_df.date_reported.dt.month
    # day = crimes_df.date_reported.dt.day
    # hour = crimes_df.date_reported.dt.hour
    # minutes = crimes_df.date_reported.dt.minute
    #
    # time_df = pd.DataFrame({'year': year,
    #                         'month': month,
    #                         'day': day,
    #                         'hour': hour,
    #                         'minute': minutes})
    #
    # df_concat = pd.concat([time_df, crimes_df], axis=1)
    # final_df = df_concat.drop('date_reported', axis=1)

    with st.beta_expander("Plot of Race"):
        race_df = crimes['victim_race'].value_counts().to_frame()
        race_df = race_df.reset_index()
        race_df.columns = ['victim_race', 'Counts']
        p01 = px.bar(race_df, x='Counts', y='victim_race')
        st.plotly_chart(p01, use_container_width=False)



    hour_to_filter = st.slider('hour', 0, 23, 17)
    fil_data = df[df['date_reported'].dt.hour == hour_to_filter]
    st.map(fil_data)


@st.cache(suppress_st_warning=True)
def SupportVector(X_train, X_test, y_train, y_test):
    # Train the model
    svc_clf = SVC(kernel='linear')
    svc_clf.fit(X_train, y_train)
    y_pred_train = svc_clf.predict(X_train)
    train_score = metrics.accuracy_score(y_train, y_pred_train) * 100
    train_r_score = metrics.recall_score(y_train, y_pred_train)
    train_p_score = metrics.precision_score(y_train, y_pred_train)
    train_f_score = metrics.f1_score(y_train, y_pred_train)
    train_report = classification_report(y_train, y_pred_train)
    train_logloss = log_loss(y_train, y_pred_train)
    train_cm = confusion_matrix(y_train, y_pred_train)

    # Test
    y_pred_test = svc_clf.predict(X_test)
    test_score = metrics.accuracy_score(y_test, y_pred_test) * 100
    test_r_score = metrics.recall_score(y_test, y_pred_test)
    test_p_score = metrics.precision_score(y_test, y_pred_test)
    test_f_score = metrics.f1_score(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test)
    test_logloss = log_loss(y_train, y_pred_train)
    test_cm = confusion_matrix(y_test, y_pred_test)

    return train_score, train_p_score, train_r_score, train_f_score, train_logloss, train_cm, train_report, \
           test_score, test_p_score, test_r_score, test_f_score, test_logloss, test_cm, test_report, svc_clf
