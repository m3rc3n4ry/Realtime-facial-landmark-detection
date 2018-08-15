# import the necessary packages
from imutils.video import VideoStream
from imutlis import face_utils
import datetime
import argparse
import imutlis
import time
import dlib
import cv2

# constructing the argument parse and parsing the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required = True,
	help = "path to facial landmark detector")
ap.add_argument("-r", "--picamera", type = int, default = -1,
	help = "wheather or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
# --shape-predictor : The path to dlib’s pre-trained facial landmark detector.
# Use the “Downloads” section of this blog post to download an archive of the
# code + facial landmark predictor file.
# --picamera : An optional command line argument, this switch indicates
# whether the Raspberry Pi camera module should be used instead of the default
# webcam/USB camera. Supply a value > 0 to use your Raspberry Pi camera.


# initialize dlib's face detector (HOG-based) and then create the facial
# landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = cv2.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the camera sensor to warm-up
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera = args["picamera"] > 0).start()
time.sleep(2.0)

# the heart of out video processing pipeline can be found inside the while loop
# below
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width = 400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)
