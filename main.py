from utils.get_subtitles import export_table_to_csv, get_relevant_characters
from utils.split_video import split_video_by_quotes

table_name = "sousou_no_frieren"
min_duration = 1.5  # seconds
max_duration = 7.0  # seconds
sample_rate = 44100  # Hz

# if need to only export the table, run this
# export_table_to_csv(table_name)

# filters the table directly on postgres to get the relevant characters
characters = get_relevant_characters(
    table_name=table_name,
    min_duration=min_duration,
    max_duration=max_duration,
    min_total_time_spoken=180.0,
)
# filters the underlying csv file and then splits the video by quotes
split_video_by_quotes(
    table_name,
    episodes=1,
    min_duration=min_duration,
    max_duration=max_duration,
    sample_rate=44100,
    characters=characters,
)
