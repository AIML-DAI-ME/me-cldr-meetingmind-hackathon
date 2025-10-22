import os
import subprocess
import tempfile
from typing import List, Tuple

from google.cloud import speech, storage


def extract_audio_ffmpeg(video_path: str, sample_rate_hz: int) -> str:
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ac",
        "1",
        "-ar",
        str(sample_rate_hz),
        "-vn",
        "-f",
        "wav",
        tmp_wav.name,
    ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return tmp_wav.name


def upload_to_gcs(local_path: str, bucket_name: str, blob_name: str) -> str:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"


def transcribe_gcs_video_with_cache(
    gcs_video_uri: str,
    google_credentials_path: str,
    sample_rate_hz: int = 16000,
    language_code: str = "en-US",
    encoding: str = "LINEAR16",
    model: str = "video",
) -> Tuple[str, List[str], bool]:
    """
    Checks if a transcription exists for a GCS video.
    If not, transcribes the video and caches the result as a .txt file in GCS.
    Returns: (full_transcript, segment_list)
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path
    storage_client = storage.Client()
    speech_client = speech.SpeechClient()

    # Parse GCS video path
    assert gcs_video_uri.startswith("gs://")
    bucket_name, blob_path = gcs_video_uri.replace("gs://", "").split("/", 1)
    video_name = os.path.basename(blob_path).rsplit(".", 1)[0]

    # Check for transcription
    transcript_blob_path = f"transcription/{video_name}.txt"
    audio_blob_path = f"audio/{video_name}.wav"
    bucket = storage_client.bucket(bucket_name)
    transcript_blob = bucket.blob(transcript_blob_path)

    if transcript_blob.exists():
        transcript = transcript_blob.download_as_text()
        return transcript, transcript.split(". "), True

    # Download video to temp
    print("⬇️ Downloading video...")
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    bucket.blob(blob_path).download_to_filename(tmp_video.name)

    # Extract audio
    print("🎧 Extracting audio...")
    wav_path = extract_audio_ffmpeg(tmp_video.name, sample_rate_hz)

    # Upload audio
    print("☁️ Uploading audio to GCS...")
    gcs_audio_uri = upload_to_gcs(wav_path, bucket_name, audio_blob_path)

    # Transcribe
    print("📝 Transcribing via long-running recognizer...")
    audio = speech.RecognitionAudio(uri=gcs_audio_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding[encoding],
        sample_rate_hertz=sample_rate_hz,
        language_code=language_code,
        model=model,
    )

    operation = speech_client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=600)

    transcripts = [result.alternatives[0].transcript for result in response.results]
    full_transcript = " ".join(transcripts)

    # Save transcript to GCS
    print("💾 Uploading transcript to GCS...")
    transcript_blob.upload_from_string(full_transcript)

    return full_transcript, transcripts, False
