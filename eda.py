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
import matplotlib.pyplot as plt
from plotly import graph_objs as go

from home_page import filtered_data

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

@st.cache(suppress_st_warning=True)
def eda_data():
	df = filtered_data()
	return df

df = eda_data()
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
	submenu = st.sidebar.selectbox("Submenu",["EDA","Date"])
	crimes = filtered_data()

	if submenu == "EDA":
		st.subheader("Information about dataset")

		with st.beta_expander("Snapshot of Data"):
			st.write(df.head().T)
			st.write('Shape of dataset', df.shape)

		with st.beta_expander("Heat Map"):
			fig, ax = plt.subplots()
			sns.heatmap(df.corr(), ax=ax)
			st.write(fig)

		with st.beta_expander("Best features"):
			X = df.iloc[:, 0:8]  # independent columns
			y = df.iloc[:, -1]  # target column i.e clsd
			bestfeatures = SelectKBest(score_func=chi2, k=5)
			fit = bestfeatures.fit(X, y)
			dfscores = pd.DataFrame(fit.scores_)
			dfcolumns = pd.DataFrame(X.columns)
			# Concat two dataframes for better visualization
			featureScores = pd.concat([dfcolumns, dfscores], axis=1)
			featureScores.columns = ["Specs", "Score"]  # naming the dataframe columns
			st.write(featureScores.sort_values(by=['Score'], ascending=False))


		with st.beta_expander("Gender"):
			victim_gender_dist = pd.DataFrame(df['victim_gender'].value_counts())
			victim_gender_dist.index = ['FEMALE', 'MALE','UNKNOWN' ]
			st.write(victim_gender_dist)
			st.bar_chart(victim_gender_dist)

			# gender = df.groupby('victim_gender').sum()
			# fig = go.Figure(
			# 	df[go.Bar(x=gender.index.values, y=gender)]
			# )
			# st.write(fig.show())



		with st.beta_expander("Race"):
			race_dist = pd.DataFrame(crimes['victim_race'].value_counts())
			st.write(race_dist)
			st.bar_chart(race_dist)

		with st.beta_expander("Age"):
			age_dist = pd.DataFrame(crimes['victim_age'].value_counts())
			st.write(age_dist)
			st.bar_chart(age_dist)

		with st.beta_expander("Close Code"):
			clsd_dist = pd.DataFrame(df['clsd'].value_counts())
			clsd_dist.index = ['NO ARREST', 'ARREST']
			st.write(clsd_dist)
			st.bar_chart(clsd_dist)



	elif submenu == "Date":
		with st.beta_expander("Geographical Map Area"):
			st.map(crimes)

		with st.beta_expander("Year"):
			year_dist = pd.DataFrame(df['year'].value_counts())
			st.line_chart(year_dist)

		with st.beta_expander("Month"):
			month_dist = pd.DataFrame(df['month'].value_counts())
			st.bar_chart(month_dist)

		with st.beta_expander("Week"):
			week_dist = pd.DataFrame(crimes['dayofweek'].value_counts())
			st.bar_chart(week_dist)

		with st. beta_expander("Hour"):
			hour_dist = pd.DataFrame(df['hour'].value_counts())
			st.area_chart(hour_dist)



