# Video Process

This is a Python project that processes video files and extracts the audio features (MFCC and Deltas) from the audio files.

## Requirements

- Python 3.10
- ffmpeg

## Usage

To use the script, you need to have the following files in the `data` folder:

- `sousou_no_frieren.csv`: This is a CSV file containing the episode information and the quotes from the anime.
  That can be generated using the export_table_to_csv function.

You can use the `main.py` file to process the video files and extract the audio features. This will created one folder per character, containing the audio files for each of its quotes.

- Next step is to input this files into UVR and get the cleaned samples.
  After this, run the exploring.py file to calculate the mcc and deltas for each character and save the average of them as an nparray (this excludes the first coefficient, which is the log of the total energy).

### Audio Format

Here's a quick walkthrough:

Signed 16 bit PCM (assuming 44.1kHz) is standard CD quality. Signed 24 bit has more dynamic range, so is 'higher quality'. If file size is no issue I would render 24-bit PCM for most things. Basically everything gets converted to one of these two for playback, so this is convenient for everyone.
In response to 'is there a better format than WAV?'. That can be a nuanced question, but in short, WAV is completely uncompressed and lossless; it is 100% of what you hear from your DAW (internally this is what the DAW is working with and sending to the soundcard). FLAC is also lossless, but has some compression so the files are smaller. It's inconvenient for production work, but some places will accept it for delivery.

## Issues

Apparently, the UVR model messes up the frequencies values, is becomes almost discrete.
Same thing happens using the VocalRemover website.
