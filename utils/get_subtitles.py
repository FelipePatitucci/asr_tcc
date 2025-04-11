import os
from pathlib import Path

import psycopg2
import polars as pl
from dotenv import load_dotenv

from .helpers import resolve_str_path

load_dotenv()
user = os.getenv("USER")
password = os.getenv("PASSWORD")
host = os.getenv("HOST")
database = os.getenv("DATABASE")
port = os.getenv("PORT")

query_data = """
    SELECT 
        mal_id,
        episode,
        name,
        quote, 
        TO_CHAR(START_TIME, 'HH24:MI:SS.MS') AS START_TIME,
        TO_CHAR(END_TIME, 'HH24:MI:SS.MS') AS END_TIME,
        extract(milliseconds from end_time - start_time) as duration_ms,
        row_number() over (order by episode, start_time) as row_idx
    FROM anime_quotes.raw_quotes.{table_name}
    ORDER BY episode, start_time
"""
query_characters = """
-- total quote time and amount of quotes by each character
with quote_metrics as (
select 
	name, 
	round(sum(duration_ms) / 1000, 2) as total_time_seconds,
	count(*) as quote_amount
from (%s) data
where name not ilike '%%LYRICS%%'
and duration_ms between %f and %f
group by name
)
select name 
from quote_metrics 
where total_time_seconds > %f
order by total_time_seconds desc;
"""
schema = {
    "mal_id": pl.UInt32,
    "episode": pl.UInt16,
    "name": pl.Utf8,
    "quote": pl.Utf8,
    "start_time": pl.Utf8,
    "end_time": pl.Utf8,
    "duration_ms": pl.UInt32,
    "row_idx": pl.UInt32,
}


def postgres_connector(
    user: str = user,
    password: str = password,
    host: str = host,
    database: str = database,
    port: str = port,
    autocommit: bool = True,
) -> psycopg2.extensions.connection:
    try:
        connection = psycopg2.connect(
            host=host, port=port, database=database, user=user, password=password
        )
        if autocommit:
            # 0 is for autocommit, try psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT
            connection.set_isolation_level(0)
    except Exception as error:
        print("Error while connecting to PostgreSQL", error)
        raise

    return connection


def fetch_data(
    query: str, connection: psycopg2.extensions.connection = None
) -> list[tuple]:
    """
    Fetch data from a PostgreSQL table.

    Parameters:
    - query: Query to be executed.
    - connection: psycopg2 database connection object.
    """
    if connection is None:
        connection = postgres_connector()

    cursor = connection.cursor()
    try:
        cursor.execute(query)
        data = cursor.fetchall()
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
    finally:
        cursor.close()

    return data


def export_table_to_csv(
    table_name: str = "sousou_no_frieren",
    csv_file_path: str | Path = None,
    connection: psycopg2.extensions.connection = None,
    schema: dict[str, pl.DataType] = schema,
) -> None:
    """
    Runs a query in the postgres database and export the results to a CSV file.

    Parameters:
    - table_name (str): Name of the table (on Postgres) to be queried.
    - csv_file_path (str): The path to the CSV file.
    - connection (psycopg2.connection): The database connection.
    - schema (dict[str, pl.DataType]): The schema of the DataFrame.
    """
    if csv_file_path is None:
        csv_file_path = resolve_str_path(f"data/{table_name}/{table_name}.csv")

    elif isinstance(csv_file_path, str):
        csv_file_path = resolve_str_path(csv_file_path)

    data = fetch_data(query_data.format(table_name=table_name), connection)
    df = pl.DataFrame(data, schema=schema, orient="row", infer_schema_length=1000)
    df.write_csv(csv_file_path)
    print(f"DataFrame exported to {csv_file_path}. {df.shape[0]} rows written.")


def get_relevant_characters(
    table_name: str = "sousou_no_frieren",
    conn: psycopg2.extensions.connection = None,
    min_duration: float = 1.5,
    max_duration: float = 7.0,
    min_total_time_spoken: float = 180.0,
) -> list[str]:
    """
    Queries the database for characters with a minimum total time spoken.
    The total time is calculated by summing the duration of all quotes by a character.
    Quotes that have a duration outside the specified range (min_duration, max_duration) are excluded.

    Parameters:
    - conn (psycopg2.connection): The database connection.
    - min_duration (float): Minimum duration of a quote in seconds. Default is 1.5 seconds.
    - max_duration (float): Maximum duration of a quote in seconds. Default is 7.0 seconds.
    - min_total_time_spoken (float): Minimum total time spoken in seconds. Default is 180.0 seconds.

    Returns:
    - list[str]: A list of relevant characters.
    """
    data = fetch_data(
        query_characters
        % (
            query_data.format(table_name=table_name),
            min_duration * 1000,
            max_duration * 1000,
            min_total_time_spoken,
        ),
        conn,
    )
    return [row[0] for row in data]


def fix_overlapping_quotes(episode_df: pl.DataFrame) -> pl.DataFrame:
    """
    Removes overlapping quotes from an episode dataframe.

    Parameters:
    - episode_df (pl.DataFrame): The episode dataframe to be cleaned.

    Returns:
    - pl.DataFrame: The cleaned dataframe.
    """
    cleaned_rows = []
    last_end_time = None
    current_episode = 0
    skipped_quote = False

    for idx, row in enumerate(episode_df.iter_rows(named=True)):
        if idx == 0:
            last_end_time = row["end_time"]
            current_episode = row["episode"]
            cleaned_rows.append(row)
            continue

        if row["start_time"] < last_end_time and row["episode"] == current_episode:
            # this means that we have overlapping quotes, hence its wiser to remove all of them
            skipped_quote = True
            continue

        if skipped_quote:
            cleaned_rows.pop()
            skipped_quote = False

        last_end_time = row["end_time"]
        current_episode = row["episode"]
        cleaned_rows.append(row)

    # Convert the cleaned rows back to a dataframe
    return pl.DataFrame(cleaned_rows, schema=episode_df.schema)


def filter_data_from_csv(
    csv_file: str | Path,
    episode_numbers: list[int] | int = 1,
    min_duration: float = 1.5,
    max_duration: float = 7.0,
    characters: list[str] | None = None,
) -> pl.DataFrame:
    """
    Filters the data from a CSV file based on episode numbers and minimum duration.

    Parameters:
    - csv_file (str): The path to the CSV file.
    - episode_numbers (list[int]): A list of episode numbers to filter by.
    - min_duration (float): The minimum duration in seconds to filter by.
    - max_duration (float): The maximum duration in seconds to filter by.
    - characters (list[str]): A list of characters to filter by. Defaults to None (all are considered).

    Returns:
    - pl.DataFrame: A DataFrame containing the filtered data.
    """
    if isinstance(csv_file, str):
        csv_file = resolve_str_path(csv_file)

    df = pl.read_csv(source=csv_file, schema=schema, has_header=True)

    if isinstance(episode_numbers, int):
        episode_numbers = [episode_numbers]

    # Filter by episode numbers and minimum duration
    filtered_df = df.filter(
        (pl.col("episode").is_in(episode_numbers))
        & (pl.col("duration_ms") >= int(min_duration * 1000))
        & (pl.col("duration_ms") <= int(max_duration * 1000))
    )

    if characters is not None:
        # optionally, filter by characters
        filtered_df = filtered_df.filter(pl.col("name").is_in(characters))

    # remove overlapping quotes
    filtered_df = fix_overlapping_quotes(filtered_df)

    return filtered_df
