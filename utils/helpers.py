from pathlib import Path


def find_folder(folder_name: str) -> Path:
    """
    Finds a folder by name in the current working directory or any parent directory.

    If the folder is not found in the current working directory, the function will
    search upwards in the directory tree until it finds the folder. If the folder is
    found, it returns a Path objects.

    Parameters:
    folder_name : str
        The name of the folder to search for.

    Returns:
    Path
        The path to the target directory.

    Raises:
    FileNotFoundError:
        If the folder is not found in the current working directory or any parent directories.
    """
    current_dir = Path.cwd()

    # Search for the folder in the current directory or upward in the tree
    for parent in [current_dir] + list(current_dir.parents):
        target_folder = parent / folder_name

        if target_folder.is_dir():
            return target_folder

    # If folder is not found, raise an error
    raise FileNotFoundError(
        f"Folder '{folder_name}' not found in the current directory or any parent directories."
    )


def resolve_str_path(path: str) -> Path:
    """
    Resolves a string path to a Path object.

    Parameters:
    - path (str): The string path to be resolved.

    Returns:
    - Path: The resolved Path object.
    """
    converted = Path(path)
    absolute_folder = find_folder(converted.parent)
    return Path.joinpath(absolute_folder, converted.name)


def cmd_clip_video(
    video_path: str | Path, start_time: str | Path, end_time: str, output_filename: str
) -> list[str]:
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
    video_path: str | Path,
    output_filename: str | Path,
    start_time: str,
    end_time: str,
    sample_rate: int = 44100,
) -> list[str]:
    """
    Convert a video file (.mkv) to a .wav audio file, change it to mono.

    Parameters:
    - video_path: str, path to the input .mkv video file
    - output_filename: str, path to the output .wav audio file
    - start_time: str, start time in the format "hh:mm:ss.xxx"
    - end_time: str, end time in the format "hh:mm:ss.xxx"
    - sample_rate: int, sample rate for the output .wav file in Hz (default is 44100 Hz)
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
