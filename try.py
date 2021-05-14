import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


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

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

def get_key(val,my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


@st.cache(suppress_st_warning=True)
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
        week = st.selectbox('DayofWeek', ('SUNDAY', 'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY'))


    geolocator = Nominatim(user_agent="GTA Lookup")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geolocator.geocode(street + ", " + city + ", " + state + ", USA")

    lat = location.latitude
    lon = location.longitude

    st.sidebar.subheader('Select datetime')
    year = st.sidebar.slider('year', 2010, 2025, 2018)
    month = st.sidebar.slider('month', 1, 12, 5)
    day = st.sidebar.slider('day', 1, 31, 23)
    hour = st.sidebar.slider('hour', 1, 24, 15)

    data = {'address':street + ", " + city + "," + state + ", USA",
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



user_df = user_input_features()
data = user_df[0]
st.subheader('Your input data: ')
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
selected_sample = [year_en, month_en, day_en, hour_en, week_en, age_en, race_en, gender_en, lat_en,
                   lon_en]
sample = np.array(selected_sample).reshape(1, -1)
st.subheader('User data')
st.write(sample)

#Map
map_data = pd.DataFrame({'lat': [lat_en], 'lon': [lon_en]})
st.map(map_data, zoom=14)
