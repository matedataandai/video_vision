import streamlit as st
from cameras import VideoRecorder
st.set_page_config(page_title="Record session", layout="wide")

# ---------------------------------------------------------------
# Data: possible lesson types available at each date/time slot
# ---------------------------------------------------------------
court_options = [
    "-- Select --",
    "Court 1",
    "Court 2",
    "Court 3"
]

court = st.selectbox(label="Court", options=court_options, label_visibility="collapsed")
email = st.text_input(label="Email - If multiple emails, please separate them with commas", label_visibility="collapsed")
duration = st.slider(label="Duration (minutes)", min_value=1, max_value=120, value=30, step=10, label_visibility="collapsed")
submit_button = st.button("Submit")
if submit_button:
    if court == "-- Select --":
        st.warning("Please select a court before submitting.")
    elif not email:
        st.warning("Please enter your email before submitting.")
    else:
        st.success(f"✅ Court **{court}** booked for **{email}** for **{duration}** minutes.")
        VideoRecorder(court=court, email=email, duration=duration).start_recording()
        st.balloons()