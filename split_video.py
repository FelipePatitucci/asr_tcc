import subprocess
import os
from datetime import timedelta
from time import time

from get_subtitles import (
    export_table_to_csv,
    get_relevant_characters,
    filter_data_from_csv,
)
from commands import cmd_clip_audio


def milliseconds_to_time_string(ms, epsilon: int = 0):
    """
    Convert milliseconds to a formatted time string of the form hh:mm:ss:xxx.

    Parameters:
    ms (float): The time in milliseconds.

    Returns:
    str: The formatted time string.
    """
    # Convert milliseconds to seconds
    seconds = ms / 1000.0
    # Create a timedelta object
    td = timedelta(seconds=seconds) - timedelta(milliseconds=epsilon)
    # Format the timedelta to a string, removing the days part and leading zeros
    time_str = str(td)
    # Splitting the timedelta string to remove the microseconds part and keep only up to milliseconds
    hours, minutes, secs = time_str.split(":")
    try:
        secs, micros = secs.split(".")
    except ValueError:
        # secs is an integer value
        micros = "000"
    # Get the first three digits of the microseconds as milliseconds
    millis = micros[:3]

    # Combine them into the final formatted string
    formatted_time_str = f"{int(hours):02}:{int(minutes):02}:{int(secs):02}.{millis}"
    return formatted_time_str


def run_ffmpeg_command(command):
    """Helper function to run an FFMPEG command."""
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to run ffmpeg:", e)
    except TypeError as e:
        print(command)
        raise e


def fix_time(time_str: str) -> str:
    """
    Fix time string to have the format HH:MM:SS.xxx.

    Parameters:
    time_str (str): The time string to be fixed.

    Returns:
    str: The fixed time string.
    """
    size = len(time_str)
    if size == 8:
        time_str += ".000"
    elif size > 12:
        excess = size - 12
        time_str = time_str[:-excess]

    return time_str


def split_video_by_quotes(
    table_name: str,
    video_path: str,
    output_folder: str,
    episodes: list[int] | int = 1,
    min_duration: float = 1.5,
    max_duration: float = 7.0,
    sample_rate: int = 44000,
    characters: list[str] | None = None,
) -> None:
    """
    Split episodes of a video into separate audio files based on a DataFrame.

    Parameters:
    table_name (str): Table name in the database containing the series data.
    video_path (str): Path to the folder containing the video files.
    output_folder (str): Path to the output folder where the audio files will be saved.
    episodes (list[int] | int): List of episode numbers to consider. Default is just the first episode.
    min_duration (float): Minumum segment duration for the quote to be considered. Default is 1.5 seconds.
    max_duration (float): Maximum segment duration for the quote to be considered. Default is 7.0 seconds.
    sample_rate (int): Sample rate of the audio files. Default is 44000 Hz.
    characters (list[str]): A list of characters to filter by. Defaults to None (all are considered).
    """

    st = time()
    total_parsed = 0
    file_path = f"data/{table_name}.csv"

    if not os.path.exists(file_path):
        export_table_to_csv(table_name, file_path)

    df = filter_data_from_csv(
        file_path, episodes, min_duration, max_duration, characters
    )

    # create folder just once to avoid I/O
    for curr_char in characters:
        curr_output_folder = os.path.join(output_folder, curr_char)
        if not os.path.exists(curr_output_folder):
            try:
                os.makedirs(curr_output_folder)
            except OSError as e:
                print(f"Error creating folder {curr_output_folder}: {e}")
                continue

    for i, row in enumerate(df.iter_rows(named=True)):
        curr_char = row["name"]

        # Get start and end times from the DataFrame (assuming times are in milliseconds)
        curr_ep = row["episode"]
        start_time = fix_time(str(row["start_time"]))
        end_time = fix_time(str(row["end_time"]))
        print(
            f"EP:{curr_ep:2d}; NAME:{curr_char:<7}; IDX:{i:3d}; {row['quote'][:15]:<15}; {start_time}; {end_time};"
        )

        if len(start_time) != 12 or len(end_time) != 12:
            # sometimes polars parses incorrectly time columns
            print(
                f"Parsed incorrectly ({start_time} or {end_time}). Skipping this quote."
            )
            continue

        output_filename = os.path.join(
            output_folder, curr_char, f"ep_{row['episode']}_segment_{i+1}.wav"
        )
        if os.path.exists(output_filename):
            continue

        # FFMPEG command to cut the video and apply any necessary filters
        command = cmd_clip_audio(
            os.path.join(video_path, f"ep_{curr_ep}.mkv"),
            output_filename,
            start_time,
            end_time,
            sample_rate,
        )
        run_ffmpeg_command(command)
        total_parsed += 1

    et = time()
    print(f"Took {et - st: .2f} seconds. Total parsed: {total_parsed}.")


table_name = "sousou_no_frieren"
video_path = "data/videos"
output_folder = "data/characters"
characters = get_relevant_characters(
    table_name=table_name,
    min_duration=1.5,
    max_duration=7.0,
    min_total_time_spoken=180.0,
)
split_video_by_quotes(
    table_name,
    video_path,
    output_folder,
    episodes=3,
    characters=characters,
)
