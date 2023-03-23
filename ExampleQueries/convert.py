import subprocess
import glob
import os

for f in glob.glob("*.wav"):
    if os.path.exists("temp.wav"):
        os.remove("temp.wav")
    subprocess.call(["ffmpeg", "-i", f, "-ar", "22050", "-ac", "1", "temp.wav"], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    subprocess.call(["mv", "temp.wav", f])