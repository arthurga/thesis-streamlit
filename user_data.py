import streamlit as st
from PIL import Image
import pickle

model = pickle.load(open('rf_crime_model.pkl', 'rb'))

st.write('Hello')
st.write(model)