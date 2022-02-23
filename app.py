import streamlit as st

# Custom imports 
from multipage import MultiPage
from pages import home, audio_upload, record, evaluate


st.set_page_config(
    page_title="Automatic Conversation Transcription",
    page_icon="🎙️",
    #layout="wide",
)

# Create an instance of the app 
app = MultiPage()

#title of the main page
st.title("🎙️ Automatic Conversation Transcription & Summarisation")
footer="""<style>
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: white;
text-align: center;
</style>
<div class="footer">
<p>Developed by Mikea Dimech</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

if "first_run" not in st.session_state:
    st.session_state["first_run"] = True
else:
    st.session_state.first_run = False

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

#set current page
if st.session_state.first_run == True:
    app.current_page("Home", home.app)
elif st.session_state.current_page == "☁️ Upload Audio":
    app.current_page("☁️ Upload Audio", audio_upload.app)
elif st.session_state.current_page == "🎙️ Record":
    app.current_page("🎙️ Record", record.app)
elif st.session_state.current_page == "📈 Evaluate":
    app.current_page("📈 Evaluate", evaluate.app)

# Add all application pages
app.add_page("☁️ Upload Audio", audio_upload.app)
app.add_page("🎙️ Record", record.app)
app.add_page("📈 Evaluate", evaluate.app)

# The main app
app.run()

