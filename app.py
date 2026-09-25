import streamlit as st
from cameras import VideoRecorder
from payment import SquarePaymentUI
import time

st.set_page_config(page_title="Record session", layout="wide",page_icon="Logos.png")

court_options = [
    "-- Select --",
    "Court 1",
    "Court 2",
    "Court 3"
]
time_options = [1,30,60,90,120]
st.title("🎾 Court Booking & Recording")

court = st.selectbox(label="Court", options=court_options)
email = st.text_input(label="Email - If multiple emails, separate with commas")
duration = st.selectbox(label="Duration (minutes)", options=time_options)

st.divider()

# Trigger payment module when selections are valid or persist booking state
if court != "-- Select --" and email:
    video_recorder = VideoRecorder(court=court, email=email, duration=duration)
    if video_recorder.test_camera() =="LIVE":
        # Render the payment UI module 
        #SquarePaymentUI(amount=10.00/60*duration, description=f"Video recording for {court} ({duration} mins)")
        st.session_state.outcome = "ACCEPTED"
        # Check if payment outcome stored in session state is successful
        if st.session_state.get("outcome") == "ACCEPTED":
            st.success(f"✅ **{court}** **{duration}** minutes recording for **{email}**, the video will be shared with you through email, once the session is finished. This email might be in your spam folder")
            video_recorder.start_recording()
            st.balloons()
    else:
        st.error("Camera is out of service")
else:
    st.info("Please select a court and provide your email address to proceed to payment.")
st.image("poweredbymatedata.png", width=400)