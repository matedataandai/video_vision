import streamlit as st
st.set_page_config(page_title="Video tennis session", layout="wide",page_icon="Logos.png")
st.title("Video Vision Tennis Session")


v = st.query_params.get("v") + ".mp4"


st.video(v)
st.image("poweredbymatedata.png", width=400)