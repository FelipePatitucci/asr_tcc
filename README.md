Create a .wav containing all the quotes from a specific character.
Associated with it, also a file containing the timestamps of the quotes.

For example, if the character is "Himmel", the .wav file should contain all the quotes from Himmel,
and the timestamps.txt file should contain the timestamps of the beggining and end of each quote.

The timestamps.txt file should be in the following format:

HimmelQuotes.txt
00:00:00.000 - 00:00:05.000
00:00:05.000 - 00:00:10.000
00:00:10.000 - 00:00:15.000
...

With this, we can easily use the site https://vocalremover.org/pt/ to remove the audio,
and then split back to quotes.

The question comes down to what you're delivering for. Match their requirements.

I've never had a master come back to me that hasn't been rendered into 5-10 different delivery formats, some of which are (or not) on your list.

Here's a quick walkthrough:

Signed 16 bit PCM (assuming 44.1kHz) is standard CD quality. Signed 24 bit has more dynamic range, so is 'higher quality'. If file size is no issue I would render 24-bit PCM for most things. Basically everything gets converted to one of these two for playback, so this is convenient for everyone.

Signed 32-bit PCM is a bit of a nonsense format. Unless specified otherwise, I would avoid this because most systems aren't used to dealing with it.

Unsigned 8bit PCM is pretty much only useful if you're doing retro game sounds or if space/RAM is extremely limited.

32-bit float is useful for delivering to other engineers in your pipeline. The main difference between this and 24-bit PCM is that the 32-bit float version is allowed to clip while in the digital domain, whereas 24bit PCM cannot.

64-bit float is probably excessive. I'm pretty sure that no-one could actually hear the difference from 32b fl. It exists as a format you can manipulate the amplitude a lot (let's say thousands of times) without hitting quantization errors. For most use cases, this isn't important so 32-bit float is fine. If you (or someone down the line) will be manipulating the amplitude A LOT then this is a sensible option, but otherwise you're probably wasting resources.

The other options are mostly irrelevant UNLESS they are specified in your list of delivery requirements. Otherwise it's pretty safe to ignore them

In response to 'is there a better format than WAV?'. That can be a nuanced question, but in short, WAV is completely uncompressed and lossless; it is 100% of what you hear from your DAW (internally this is what the DAW is working with and sending to the soundcard). FLAC is also lossless, but has some compression so the files are smaller. It's inconvenient for production work, but some places will accept it for delivery.

TL;DR: Do what the client asks. If you don't know which to render, 16 or 24 bit PCM WAVs are pretty universal. If you're delivering to another AE ask, but 32-bit float would be my default. Those are, by far, the three most important.

"-af",
f"lowpass=f={cutoff_frequency}", # Apply a low-pass filter with cutoff at 5kHz by default

Apparently, the UVR model messes up the frequencies values, is becomes almost discrete.
Same thing happens using the VocalRemover website.
