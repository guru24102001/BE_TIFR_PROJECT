# importing required libraries
import cv2
import time
from threading import Thread  # library for implementing multi-threaded processing
import queue
import os
# defining a helper class for implementing multi-threaded processing


class WebcamStream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id   # default is 0 for primary camera
        self.q = queue.Queue()

        # opening video capture stream
        self.vcap = cv2.VideoCapture(self.stream_id)
        # self.vcap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        if self.vcap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.vcap.read()
        if self.grabbed is False:
            print('[Exiting] No more frames to read')
            exit(0)

        # self.stopped is set to False when frames are being read from self.vcap stream
        self.stopped = True

        # reference to the thread for reading next available frame from input stream
        self.t = Thread(target=self.update, args=())
        # daemon threads keep running in the background while the program is executing
        self.t.daemon = True

    # method for starting the thread for grabbing next available frame in input stream
    def start(self):
        self.stopped = False
        self.t.start()

    # method for reading next frame
    def update(self):
        while True:
            ret, frame = self.vcap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                    # print('emptying the queue')
                except queue.Empty:
                    print("empty")
                    pass
            self.q.put(frame)
            # print(f'Length:{self.q.qsize()}')
            # print("incoming frame put in queue")
        # while True:
        #     if self.stopped is True:
        #         break
        #     self.grabbed, self.frame = self.vcap.read()
        #     if self.grabbed is False:
        #         print('[Exiting] No more frames to read')
        #         self.stopped = True
        #         break
        # self.vcap.release()

    # method for returning latest read frame
    def read(self):
        # print('waiting')
        frame = self.q.get()
        # print("Got frame")
        return frame
        # return self.frame

    # method called to stop reading frames
    def stop(self):
        self.stopped = True


# initializing and starting multi-threaded webcam capture input stream
# stream_id = 0 is for primary camera
webcam_stream = WebcamStream(
    stream_id='rtsp://Guru:Guru1@192.168.167.102:8080/h264_pcm.sdp')
webcam_stream.start()

# processing frames in input stream
num_frames_processed = 0
start = time.time()
while True:
    if webcam_stream.stopped is True:
        break
    else:
        frame = webcam_stream.read()
# replace next 2 lines with YOLO processing
    # adding a delay for simulating time taken for processing a frame
        delay = 1  # delay value in seconds. so, delay=1 is equivalent to 1 second
        

        

        cv2.imshow('frame', frame)
        cv2.imwrite("frame_folder/frame"+str(num_frames_processed)+".jpg", frame)
        time.sleep(delay)
        num_frames_processed += 1
        os.remove("frame_folder/frame"+str(num_frames_processed-1)+".jpg")
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
end = time.time()
webcam_stream.stop()  # stop the webcam stream

# printing time elapsed and fps
elapsed = end-start
fps = num_frames_processed/elapsed
print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(
    fps, elapsed, num_frames_processed))

# closing all windows
cv2.destroyAllWindows()
