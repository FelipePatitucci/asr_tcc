def cmd_clip_video(
    video_path: str, start_time: str, end_time: str, output_filename: str
):
    """Clip a video using FFMPEG."""
    return [
        "ffmpeg",
        "-i",
        video_path,
        "-ss",
        start_time,  # Start time
        "-to",
        end_time,  # End time
        "-map",
        "0:0",  # Map video stream
        "-map",
        "0:1",  # Map audio stream (the first stream)
        "-c:v",
        "copy",  # Copy video codec
        "-c:a",
        "aac",  # Transcode audio to AAC
        "-hide_banner",
        "-loglevel",
        "16",  # Quiet the output
        output_filename,
    ]


def cmd_clip_audio(
    video_path: str,
    output_filename: str,
    start_time: str,
    end_time: str,
    sample_rate: int = 44000,
):
    """
    Convert a video file (.mkv) to a .wav audio file, change it to mono.

    Parameters:
    - video_path: str, path to the input .mkv video file
    - output_filename: str, path to the output .wav audio file
    - start_time: str, start time in the format "hh:mm:ss.xxx"
    - end_time: str, end time in the format "hh:mm:ss.xxx"
    - sample_rate: int, sample rate for the output .wav file in Hz (default is 44000 Hz)
    - cutoff_frequency: int, frequency cutoff for the low-pass filter (default is 5000 Hz)
    """
    return [
        "ffmpeg",
        "-i",
        video_path,  # Input video file
        "-ss",
        start_time,  # Start time
        "-to",
        end_time,  # End time
        "-vn",  # No video (extract only audio)
        "-ac",
        "1",  # Convert to mono (1 audio channel)
        "-ar",
        str(sample_rate),  # Set the sample rate to 44100 Hz (standard for WAV)
        "-acodec",
        "pcm_s16le",  # Set the audio codec to PCM signed 16-bit little endian (standard for WAV),
        "-hide_banner",
        "-loglevel",
        "16",  # Quiet the output
        output_filename,  # Output .wav file
    ]
