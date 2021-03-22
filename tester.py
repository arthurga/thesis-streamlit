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