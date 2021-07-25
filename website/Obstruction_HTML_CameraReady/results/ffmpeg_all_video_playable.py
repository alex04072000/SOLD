import os
import glob

all_mp4 = sorted(glob.glob('video_full_reflection/**/*_Liu2020.mp4'))

for mp4 in all_mp4:
    print(mp4)
    os.system('ffmpeg -y -i "'+mp4+'" "'+mp4[:-4]+'_ffmpeg.mp4"')
