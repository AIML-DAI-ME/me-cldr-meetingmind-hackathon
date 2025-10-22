import os
import json
import time
import tempfile
import subprocess
from typing import List, Tuple

import boto3
from botocore.exceptions import ClientError


# ----------------------------- #
#       Utility Functions       #
# ----------------------------- #

def extract_audio_ffmpeg(video_path: str, sample_rate_hz: int = 16000) -> str:
    """Extracts mono audio from a video file using FFmpeg."""
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", str(sample_rate_hz),
        "-vn",
        "-f", "wav",
        tmp_wav.name,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp_wav.name


def upload_to_s3(local_path: str, bucket_name: str, key: str, region_name: str = "us-west-2") -> str:
    """Uploads a file to S3 and returns its s3:// URI."""
    s3 = boto3.client("s3", region_name=region_name)
    s3.upload_file(local_path, bucket_name, key)
    return f"s3://{bucket_name}/{key}"



def download_from_s3(bucket_name: str, key: str, region_name: str = "us-west-2") -> str:
    """Downloads a file from S3 and returns its local path."""
    s3 = boto3.client("s3", region_name=region_name)
    tmp_file = tempfile.NamedTemporaryFile(suffix=os.path.splitext(key)[-1], delete=False)
    s3.download_file(bucket_name, key, tmp_file.name)
    return tmp_file.name


# ----------------------------- #
#        Cache Management       #
# ----------------------------- #

def check_cache(bucket_name: str, transcript_key: str, region_name: str = "us-west-2") -> Tuple[bool, str]:
    """Checks if transcript cache exists in S3."""
    s3 = boto3.client("s3", region_name=region_name)
    try:
        s3.head_object(Bucket=bucket_name, Key=transcript_key)
        obj = s3.get_object(Bucket=bucket_name, Key=transcript_key)
        transcript = obj["Body"].read().decode("utf-8")
        return True, transcript
    except ClientError:
        return False, ""


def save_transcript_cache(bucket_name: str, key: str, transcript: str, region_name: str = "us-west-2"):
    """Saves plain text transcript to S3 for caching."""
    s3 = boto3.client("s3", region_name=region_name)
    s3.put_object(Bucket=bucket_name, Key=key, Body=transcript)


# ----------------------------- #
#      Transcription Logic      #
# ----------------------------- #

def run_transcription_job(
    transcribe_client,
    job_name: str,
    s3_audio_uri: str,
    bucket_name: str,
    language_code: str = "en-US",
) -> str:
    """Starts a transcription job and waits for it to complete."""
    output_key = f"transcription/{job_name}.json"

    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": s3_audio_uri},
        MediaFormat="wav",
        LanguageCode=language_code,
        OutputBucketName=bucket_name,
        OutputKey=output_key,
    )

    # Wait for job to complete
    while True:
        status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        job_status = status["TranscriptionJob"]["TranscriptionJobStatus"]
        if job_status in ["COMPLETED", "FAILED"]:
            break
        print("⏳ Waiting for transcription to finish...")
        time.sleep(10)

    if job_status == "FAILED":
        raise RuntimeError(f"Transcription job failed: {status}")

    return output_key


# ----------------------------- #
#       Main Orchestrator       #
# ----------------------------- #

def transcribe_s3_video_with_cache(
    s3_video_uri: str,
    region_name: str = "us-west-2",
    sample_rate_hz: int = 16000,
    language_code: str = "en-US",
) -> Tuple[str, List[str], bool]:
    """
    Transcribes an S3 video with caching.
    Returns: (full_transcript, segments, is_cached)
    """
    # Parse S3 video URI
    assert s3_video_uri.startswith("s3://"), "Invalid S3 URI format"
    bucket_name, key = s3_video_uri.replace("s3://", "").split("/", 1)
    video_name = os.path.basename(key).rsplit(".", 1)[0]

    transcript_key = f"transcription/{video_name}.txt"
    audio_key = f"audio/{video_name}.wav"

    # 1️⃣ Check cache
    is_cached, cached_transcript = check_cache(bucket_name, transcript_key, region_name)
    if is_cached:
        print("✅ Using cached transcript")
        return cached_transcript, cached_transcript.split(". "), True

    # 2️⃣ Download video
    print("⬇️ Downloading video from S3...")
    tmp_video = download_from_s3(bucket_name, key, region_name)

    # 3️⃣ Extract and upload audio
    print("🎧 Extracting audio...")
    wav_path = extract_audio_ffmpeg(tmp_video, sample_rate_hz)

    print("☁️ Uploading audio to S3...")
    s3_audio_uri = upload_to_s3(wav_path, bucket_name, audio_key, region_name)

    # 4️⃣ Transcribe
    transcribe_client = boto3.client("transcribe", region_name=region_name)
    job_name = f"transcribe-{video_name}-{int(time.time())}"
    print(f"📝 Starting Amazon Transcribe job: {job_name}")

    output_json_key = run_transcription_job(
        transcribe_client, job_name, s3_audio_uri, bucket_name, language_code
    )

    # 5️⃣ Download and parse transcript JSON
    tmp_json = download_from_s3(bucket_name, output_json_key, region_name)
    with open(tmp_json, "r") as f:
        data = json.load(f)
        full_transcript = data["results"]["transcripts"][0]["transcript"]

    # 6️⃣ Save plain text transcript to S3
    print("💾 Caching transcript to S3...")
    save_transcript_cache(bucket_name, transcript_key, full_transcript, region_name)

    return full_transcript, full_transcript.split(". "), False
