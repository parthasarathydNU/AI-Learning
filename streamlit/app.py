# This notebook will just be about trying out the different 
# Streamlit chat elements 
# Tutorial: https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st
import numpy as np


# Every time we update any element / input 
# this entire script is run from top to bottom

# Paste a normal string text - could be the header
st.write("Rubric Generator")


# Defining and updating inline chate elements
# Defining a User element
with st.chat_message("user"):
    st.write("Hello 👋")

# Defining a Bot element
with st.chat_message("assistant"):
    st.write("Hello human")
    st.bar_chart(np.random.randn(30, 3))

# Another way where we can save the reference of 
# this chat element and update it at a later point in time
message = st.chat_message("assistant")
message.write("Hello human")
message.bar_chart(np.random.randn(30, 3))


# There can only be 1 chat_input widget 
prompt = st.chat_input("Say something")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")
