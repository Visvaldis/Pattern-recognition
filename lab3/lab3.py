import cv2
import numpy as np
from numpy import newaxis
import time
import pafy

def show_video(capture):
    flag, frame = capture.read()
    while True:
        flag, frame = capture.read()
        if flag == 0:
            break
        cv2.imshow("Video", frame)
        key_pressed = cv2.waitKey(10)    #Escape to exit
        if key_pressed == 27:
            break

def get_frames_array(capture):
    flag, frame = capture.read()
    frames = frame[newaxis, ...]
    while True:
        flag, frame = capture.read()
        if flag != 0:
            frames = np.append(frames, frame[newaxis, ...], axis=0)
        else:
            break
    return frames

def get_video_from_youtube(url):
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")

    capture = cv2.VideoCapture()
    capture.open(best.url)
    return capture

#local file
file = 'banana.mp4'

#youtube video url
#url = "https://www.youtube.com/watch?v=zsQTq0eE0Sc"
#youtube stream url
url = "https://www.youtube.com/watch?v=5qap5aO4i9A"

res = np.array([])
print("--- get_frames_array() ---")
start_time = time.time()

capture = cv2.VideoCapture(file)
#capture = get_video_from_youtube(url)

#res = get_frames_array(capture)
show_video(capture)

print("--- %s seconds ---" % (time.time() - start_time))

print(res.shape)