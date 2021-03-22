# Core Packages
import streamlit as st
import streamlit.components.v1 as components
import json

# EDA Packages
import pandas as pd
import numpy as np
import base64
import time
from datetime import datetime, timedelta
from PIL import Image

# Data Viz Packages
import matplotlib
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
import missingno as msno
import plotly
import plotly.graph_objs as go
import plotly.express as px

from plotly import graph_objs as go

from home_page import filtered_data


st.markdown(
	"""
	<style>
	.main {
	background-color: #F5F5F5;
	}
	</style>
	""",
	unsafe_allow_html=True
)




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





def count_plot(dataframe, column_name, title =None, hue = None):
    '''
    Function to plot seaborn count plot
    Input: Dataframe name that has to be plotted, column_name that has to be plotted, title for the graph
    Output: Plot the data as a count plot
    '''
    base_color = sns.color_palette()[0]
    sns.countplot(data = dataframe, x = column_name, hue=hue)
    plt.title(title)
    pass



def run_eda():
	st.title("Exploratory Data Analysis")
	submenu = st.sidebar.selectbox("Submenu",["EDA","Plots"])
	crimes = filtered_data()

	if submenu == "EDA":
		c1,c2 = st.beta_columns(2)
		with c1:
			with st.beta_expander("Victim Age Distribution"):
				st.dataframe(crimes['victim_age'].value_counts())
		with c2:
			with st.beta_expander("Victim Race Distribution"):
				st.dataframe(crimes['victim_race'].value_counts())

		v1, v2 = st.beta_columns(2)
		with v1:
			with st.beta_expander("Victim Gender Distribution"):
				st.dataframe(crimes['victim_gender'].value_counts())
		with v2:

			with st.beta_expander("Day of Week Distribution"):
				st.dataframe(crimes['dayofweek'].value_counts())


	elif submenu == "Plots":
		st.subheader("Plotting")

		with st.beta_expander("Bar Chart - Gender"):
			victim_gender_dist = pd.DataFrame(crimes['victim_gender'].value_counts())
			st.bar_chart(victim_gender_dist)


		with st.beta_expander("Plot of Close Code"):
			clsd_df = crimes['clsd'].value_counts().to_frame()
			clsd_df = clsd_df.reset_index()
			clsd_df.columns = ['Category', 'Counts']
			p01 = px.pie(clsd_df, names='Category', values='Counts')
			st.plotly_chart(p01, use_container_width=True)


		with st.beta_expander("Plot of Offense"):
			offense_dist = pd.DataFrame(crimes['ucr_group'].value_counts())
			st.bar_chart(offense_dist)


		with st.beta_expander("Plot of Race"):
			race_df = crimes['victim_race'].value_counts().to_frame()
			race_df = race_df.reset_index()
			race_df.columns = ['victim_race', 'Counts']
			p01 = px.bar(race_df, x='Counts', y='victim_race')
			st.plotly_chart(p01, use_container_width=True)


		with st.beta_expander("Age"):
			age_dist = pd.DataFrame(crimes['victim_age'].value_counts())
			st.bar_chart(age_dist)

