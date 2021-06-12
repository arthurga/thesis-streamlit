# Core Packages
import streamlit as st
import streamlit.components.v1 as components
from sodapy import Socrata
import json
import requests


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
import plotly
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import folium


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

    time_df = pd.DataFrame({'year': year,
                            'month': month,
                            'day': day,
                            'hour': hour})

    df_concat = pd.concat([time_df, crimes_df], axis=1)
    final_df = df_concat.drop('date_reported', axis=1)

    final_df = final_df.dropna()
    final_df.drop_duplicates()
    final_df = final_df.reset_index(drop=True)

    return final_df

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
maps = st.beta_container()


def run_home_page():
    original_df = load_data()
    filtered_df = filtered_data()

    property_crimes = ['BURGLARY/BREAKING ENTERING', 'THEFT', 'UNAUTHORIZED USE']
    crime_selection = original_df[(original_df.ucr_group.isin(property_crimes))]


    with header:
        st.title('Neighborhood Safety Prediction App')

        bgcolor = st.color_picker("")
        st.write('Property Crimes')
        day_text = " ".join(crime_selection['ucr_group'].tolist())
        mywordcloud = WordCloud(background_color=bgcolor).generate(day_text)
        fig = plt.figure()
        plt.imshow(mywordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(fig)

        st.markdown("""
                This web application analyzes the safety of city of Cincinnati neighborhoods based on historical records of property crimes within the Cincinnati Metro area.
                The end product provides a 'safety score' for a specific location at the time of inquiry.
                Using geo-coded address as input data from the end user along with demographic information such as age, gender and race, the app translates the likelihood of an arrest at the location
                as a measure of actual and perceived safety at the location for the user. The main goal of this project is protect users from becoming victims as opposed to catching a repeat offender.

                * **Python libraries:** base64, pandas, streamlit
                * **Data source:** [data.cincinnati-oh.gov](https://data.cincinnati-oh.gov/Safety/PDI-Police-Data-Initiative-Crime-Incidents/k59e-2pvf).
                * **Part I Crime Offense:** THEFT, BURGLARY/BREAKING ENTERING, UNAUTHORIZED USE
                """)


    with dataset:
        st.header('City of Cincinnati Crime Data')
        st.write(original_df.head())
        st.write(original_df.shape)
        st.write('Original data dimensions: ' + str(original_df.shape[0]) + ' rows and ' + str(
            crime_selection.shape[1]) + ' columns')

        st.subheader('Crime Type Distribution on Property Crime Dataset')
        crime_type_dist = pd.DataFrame(crime_selection['ucr_group'].value_counts())
        st.write(crime_type_dist)


        crime_graph = px.bar(
            crime_type_dist,
            x=crime_type_dist.index,
            y='ucr_group',
            color=crime_type_dist.index)
        st.plotly_chart(crime_graph)


    with features:
        st.header('Selected Features for Further Engineering')
        st.write(filtered_df.head().T)
        st.write(filtered_df.shape)
        st.write('Filtered data dimensions: ' + str(filtered_df.shape[0]) + ' rows and ' + str(
            filtered_df.shape[1]) + ' columns')


    with maps:
        st.header('Map of Crime Incidents')
        st.map(filtered_df)
