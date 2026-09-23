import streamlit as st
st.set_page_config(page_title="Video tennis session", layout="wide",page_icon="Logos.png")
st.title("Video Vision Tennis Session")

video_map ={"output":"fixed_output.mp4"}

v = st.query_params.get("v")

if v in video_map:
    st.video(video_map["output"])
else :
    st.error("Invalid link.")
st.image("poweredbymatedata.png", width=400)