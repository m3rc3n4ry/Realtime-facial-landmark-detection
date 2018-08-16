# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
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
detector = dlib.get_frontal_face_detector()
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
	frame = imutils.resize(frame, width = 500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)
# On Line 40 we start an infinite loop that we can only break out of if we
# decide to exit the script by pressing 'q' key on the keyboard

# Line 44 grabs the next frame from our video stream

# We then preprocess this frame by resizing it to have a width of 400 pixels
# and convert it to grayscale (Lines 45 an 46).

# Before we can detect facial landmarks in our frame, we first need to localize
# the face — this is accomplished on Line 49 via the detector  which returns
# the bounding box (x, y)-coordinates for each face in the image.



# Now that we have detected the faces in the video stream, the next step is to
# apply the facial landmark predictor to each face ROI:

	#loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then convert the
		# facial landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# loop over the (x, y)-coordinates for the facial landmarks and draw
		# them on the image
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

	#show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	#if the 'q' key was pressed, break from the loop
	if key == ord("q"):
		break
# on Line 68 we loop over each of the detected faces

# Line 71 applies the facial landmark detector to the face region, returning a
# shape  object which we convert to a NumPy array (Line 72).

# Lines 76 and 77 then draw a series of circles on the output frame,
# visualizing each of the facial landmarks. To understand what facial region
# (i.e., nose, eyes, mouth, etc.) each (x, y)-coordinate maps to

# Lines 80 and 81 display the output frame  to our screen. If the q  key is
# pressed, we break from the loop and stop the script (Lines 84 and 85).


# Finally, Lines 100 and 101 do a bit of cleanup:
cv2.destroyAllWindows()
vs.stop()