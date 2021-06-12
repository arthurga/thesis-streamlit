# Core Packages
import streamlit as st
import streamlit.components.v1 as components


# Import pages
from home_page import run_home_page
from eda import run_eda
from ml import run_ml

def main():

    # Setup - PAGES
    st.sidebar.title("Menu")
    menu = ["Home", "Exploration", "Model Building", "About"]
    page = st.sidebar.radio("Please select a page", menu)

    if page == "Home":
        run_home_page()

    elif page == "Exploration":
        run_eda()

    elif page == "Model Building":
        run_ml()


    else:
        st.subheader('About Me')
        st.markdown('Name: GIFTY A ARTHUR')
        st.markdown('LinkedIn: https://www.linkedin.com/in/giftyarthur/')
        st.markdown('Github: https://github.com/arthurga')


if __name__ == "__main__":
    main()