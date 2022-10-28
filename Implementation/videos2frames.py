from moviepy.editor import VideoFileClip
import numpy as np
import os
from datetime import timedelta
# i.e if video of duration 30 seconds, saves 10 frame per second = 300 frames saved in total
SAVING_FRAMES_PER_SECOND = 1


def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def main(video_file):
    # load the video clip
    video_clip = VideoFileClip(video_file)
    # make a folder by the name of the video file
    filename = "frames"
    if not os.path.isdir(filename):
        os.mkdir(filename)
    file = open('images.txt', 'w')

    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(video_clip.fps, SAVING_FRAMES_PER_SECOND)
    # if SAVING_FRAMES_PER_SECOND is set to 0, step is 1/fps, else 1/SAVING_FRAMES_PER_SECOND
    step = 1 / video_clip.fps if saving_frames_per_second == 0 else 1 / \
        saving_frames_per_second
    # iterate over each possible frame
    for current_duration in np.arange(0, video_clip.duration, step):
        # format the file name and save it
        frame_duration_formatted = format_timedelta(
            timedelta(seconds=current_duration)).replace(":", "-")
        frame_filename = os.path.join(
            filename, f"frame{frame_duration_formatted}.jpg")
        file.write(f"frames/frame{frame_duration_formatted}.jpg\n")
        # save the frame with the current duration
        video_clip.save_frame(frame_filename, current_duration)
    file.close()


main('final_633afe3d9c1a02006f0281db_600923.mp4')
