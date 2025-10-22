import os
import tempfile
import time

import streamlit as st

from agent_launch import crew_launch, transcript_to_text
from scripts.utils_aws import upload_to_s3

st.set_page_config(page_title="MeetingMind Crew", layout="wide")
st.title("🤖📋 MeetingMind AI Assistant")

st.markdown(
    """
    Upload a meeting video or provide a s3 URI to automatically extract a transcript, summarize it, extract action items, and generate a professional meeting email.
    """
)

# --- Input Section ---
st.subheader("📥 Input Meeting Source")

s3_uri = st.text_input(
    "s3 Video URI (optional)", placeholder="s3://your-bucket/video.mp4"
)

uploaded_file = st.file_uploader(
    "Or upload a local video", type=["mp4", "mov", "mkv"], label_visibility="visible"
)


start_button = st.button("🚀 Start Meeting Analysis")

# --- Processing ---
if start_button:

    if not s3_uri and not uploaded_file:
        st.error("❌ Please provide either a s3 URI or upload a video file.")
        st.stop()

    with st.spinner("🔄 Processing input..."):
        # If local file uploaded, upload to s3 first
        if uploaded_file:
            # Use original filename instead of tempfile name
            original_filename = uploaded_file.name
            tmp_path = os.path.join(tempfile.gettempdir(), original_filename)

            # Write file to temp directory using original name
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())

            bucket_name = "meeting-mind-library"
            key = f"video/{original_filename}"

            st.info("Uploading video to s3...")
            s3_uri = upload_to_s3(tmp_path, bucket_name, key)
            st.success(f"✅ Uploaded to s3: `{s3_uri}`")


        # Step 1: Transcribe
        st.info("Transcribing video...")
        transcript, is_cached = transcript_to_text(s3_uri)

        if is_cached:
            time.sleep(15)

        st.text_area("📄 Transcript Preview", transcript, height=200)

        # Step 2: Launch Crew
        st.info("Running CrewAI agents...")
        result = crew_launch(transcript)

    # --- Output ---
    st.success("✅ Crew execution completed!")

