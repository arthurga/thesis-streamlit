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
from impyute.imputation.cs import mice

# Data Viz Packages
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from wordcloud import WordCloud

# Modeling Packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

    crimes = crimes_df.rename(columns={'longitude_x': 'lon', 'latitude_x': 'lat'})

    crimes['lon'] = crimes['lon'].astype(float)
    crimes['lat'] = crimes['lat'].astype(float)

    crimestats = crimes

    return crimestats


# Filtered data
@st.cache(suppress_st_warning=True)
def filtered_data():
    data_df = load_data()
    property_crimes = ['BURGLARY/BREAKING ENTERING', 'THEFT', 'UNAUTHORIZED USE']
    crime_years = list(range(2010, 2021))
    crimes = data_df[(data_df.date_reported.dt.year.isin(crime_years)) & (data_df.ucr_group.isin(property_crimes))]

    features = ['date_reported', 'dayofweek', 'victim_age', 'victim_race', 'victim_gender', 'lon', 'lat', 'ucr_group','clsd']
    crimes_df = crimes[features]

    # Breaking down date_reported into individual features
    year = crimes_df.date_reported.dt.year
    month = crimes_df.date_reported.dt.month
    day = crimes_df.date_reported.dt.day
    hour = crimes_df.date_reported.dt.hour
    minutes = crimes_df.date_reported.dt.minute

    time_df = pd.DataFrame({'year': year,
                            'month': month,
                            'day': day,
                            'hour': hour,
                            'minute': minutes})

    df_concat = pd.concat([time_df, crimes_df], axis=1)
    final_df = df_concat.drop('date_reported', axis=1)
    final_df = final_df.dropna()
    final_df.drop_duplicates()
    final_df = final_df.reset_index(drop=True)

    return final_df

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()


def run_home_page():
    original_df = load_data()
    filtered_df = filtered_data()

    property_crimes = ['BURGLARY/BREAKING ENTERING', 'THEFT', 'UNAUTHORIZED USE']
    crime_selection = original_df[(original_df.ucr_group.isin(property_crimes))]


    with header:
        st.title('Neighborhood Safety Prediction App')

        bgcolor = st.beta_color_picker("")

        with st.beta_expander("Property Crimes", expanded=True):
            day_text = " ".join(crime_selection['ucr_group'].tolist())
            mywordcloud = WordCloud(background_color=bgcolor).generate(day_text)
            fig = plt.figure()
            plt.imshow(mywordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig)

        st.markdown("""
                This web application analyzes the safety of city of Cincinnati neighborhoods based on historical records of property crimes within the Cincinnati Metro area.
                The app builds a predictive model by using historical crime data on victims of Property Crimes from the official portal of the Cincinnati Police Department.
                The end product provides a Safety Score for a specific location at the time of inquiry.
                Using geocoded address as input data from the end user along with biodata such as age, gender and race, the app translates the likelihood of an arrest at the location
                as a measure of safety at the location and its surrounding areas for the user. The main goal of this project is to make the crime analytic information readily available to end users for better decision-making.

                * **Python libraries:** base64, pandas, streamlit
                * **Data source:** [data.cincinnati-oh.gov](https://data.cincinnati-oh.gov/Safety/PDI-Police-Data-Initiative-Crime-Incidents/k59e-2pvf).
                * **Part I Crime Offense:** THEFT, BURGLARY/BREAKING ENTERING, UNAUTHORIZED USE
                """)


    with dataset:
        st.header('City of Cincinnati Crime Data')
        st.write(original_df.head())
        st.write('Original data dimensions: ' + str(original_df.shape[0]) + ' rows and ' + str(
            original_df.shape[1]) + ' columns')

        st.subheader('Victim Gender Distribution on Crime Dataset')
        victim_race_dist = pd.DataFrame(original_df['victim_gender'].value_counts()).head(10)
        st.bar_chart(victim_race_dist)

    with features:
        st.header('Selected Features for Further Engineering')
        st.write(filtered_df.head().T)
        st.write('Filtered data dimensions: ' + str(filtered_df.shape[0]) + ' rows and ' + str(
            filtered_df.shape[1]) + ' columns')
