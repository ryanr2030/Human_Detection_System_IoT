import time
import json
import cv2
import numpy as np
from scipy import ndimage
import sys
import os
import picamera
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera



def detector(args):
    # CLASSIFIER command line switches
    print(args)
    if (args[1] == "haar" or args[1] == "Haar"):
        classifier = cv2.CascadeClassifier(
            "/home/pi/Desktop/project_master/input_files/classifiers/haarcascade_fullbody.xml")
    elif (args[1] == "hog" or args[1] == "Hog"):
        classifier = cv2.HOGDescriptor()
        classifier.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    elif (args[1] == "yolo" or args[1] == "Yolo"):
        # ToDo set up yolo detector
        classifier = ''
    else:
        print("Invalid command: Please enter a command in the form of either\n"
              "detector-notPi.py detectionMethod cam gt PATHtoGroundTruthFile noshow\n"
              "detector-notPi.py detectionMethod vid PATHtoInputVid gt PATHtoGroundTruthFile noshow\n"
              "gt switch can be ommitted and noshow can be ommitted to display the detections on the monitor")
        exit(1)

    stats = open("/home/pi/Desktop/project_master/output_files/stats.txt", "w")

    class groundTruth:
        def __init__(self, type, height, width, x, y, frameN):
            self.type = type
            self.height = height
            self.width = width
            self.x = x
            self.y = y
            self.frameN = frameN
            self.true = 0

    groundT = {}
    cur = 1
    tokens = ''
    words = {}
    detection =[]
    file=None
    # GROUNTRUTH command line switches
    if (len(args)>3 and args[3] == "gt"):
        if (args[4] is None):
            file=None;
        else:
            file = args[4]
    elif (len(args)>5 and args[4] == "gt"):
        if (len(args)<6):
            file = open('/home/pi/Desktop/project_master/input_files/ground_truth/GroundTruth.csv',
                'r')
        else:
            file = open(args[5])
    if (file is not None):
        file = file.readlines()
        for lines in file[0:]:
            lines = lines.strip("\r\n")
            lines = lines.strip('"')
            words = lines.split(',')
            groundT[cur] = groundTruth(words[0], words[1], words[2], words[3], words[4], words[5])
            cur += 1
        print(groundT[1].x)

    # VIDEO INPUT command line switches
    # Apply detector on the devices default camera
    if (args[2] == 'cam' or args[2] == 'Cam'):
        if(args[1]=='hog' or args[1]=='Hog'):
            camera = PiCamera()
            rawCapture = PiRGBArray(camera,)
            camera.capture(rawCapture, format="bgr")
        else:
            cam= VideoStream(usePiCamera=True).start()
    # Apply detector to a specified input video
    elif (args[2] == 'vid' or args[2] == 'Vid'):
        cam = cv2.VideoCapture(args[3])
    else:
        print("Input error: must specify cam to use default device or vid PATHtoVid to apply on a video file")
    # set height and width (Input width and height must match output width and height)
    if (args[2] == 'cam' or args[2] == 'Cam'):
        print()
    else:
        cam.set(3, 640)
        cam.set(4, 480)

    # Create the output writer with the correct codec
    # !!!!!MINA BOTROS THIS IS WHERE THE OUTPUT VIDEO GETS WRITTEN CHANGE NAME IF YOU WANT TO SAVE THE VIDEO!!!!!!
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter('/home/pi/Desktop/project_master/output_files/output_vids/Haar_Vid2.mp4', fourcc, 2, (480, 640))

    # Frame count variable for keeping track of people in each frame
    # will be used for tracking
    framecount = 1
    start_time = time.time()
    last_upload_time=start_time
    upload_timer = start_time
    while (True):
        # computer reads the image
        if (args[1] == 'Haar' or args[1] == 'haar'):
            frame = cam.read()
        elif (args[1]=='Hog' or args[1]=='hog'):
            frame = rawCapture.array
        elif(args[1]=='Yolo' or args[1]=='Yolo'):
            #ToDo: Enter yolo frame intialization
            print()
        i = 1

        key = cv2.waitKey(1)

        #Draw the ground truth boxes if supplied
        if(len(args)>4 and (args[4]=="gt" or args[3]=="gt")):
            while i < cur:
                if str(framecount) == groundT[i].frameN and groundT[i].y != '':
                    cv2.rectangle(frame, (int(float(groundT[i].x)), int(float(groundT[i].y))), (
                        int(float(groundT[i].x) + float(groundT[i].width)),
                        int(float(groundT[i].y) + float(groundT[i].height))), (255, 255, 255), 2)
                i += 1



        # Initialize the detector with the correct classifier model
        if (args[1] == "hog" or args[1] == "Hog"):
            (pedestrian, weights) = classifier.detectMultiScale(frame)
        elif (args[1] == "haar" or args[1] == "Haar"):
            pedestrian = classifier.detectMultiScale(frame)
        elif (args[1] == "yolo" or args[1] == "Yolo"):
            # ToDo add yolo classifier initilization
            pedestrian = ""

        # Counts the number of people in each frame
        count = 0

        # Apply the classifier to each frame and draws a rectangle around the detected people
        # and formats the detection into a transmittable packet
        max = 0
        for (x, y, w, h) in pedestrian:
            k = 2
            # count the number of people
            count += 1

            # draw the rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            global t
            t = 1
            # store the center of each person
            he = int(x + w / 2)
            wi = int(y + h / 2)

            # JSON format for transmission
            detection.append("{Human: frame: " + str(framecount) + ", num_in_frame: " + str(
                count) + ", x: " + str(he) + ", y: " + str(wi) + "}")
            #individual detections to be stored locally as a local log for debugging purposes
            location = str(framecount) + ' ' + str(count) + ' ' + str(x) + ' ' + str(y) + ' ' + str(x + w) + ' ' + str(
                y + h)

            # draw a circle at the center of the person with their centers
            cv2.circle(frame, (he, wi), 2, (255, 0, 0), 2)
            cv2.putText(frame, "(" + str(x) + "," + str(y) + ")", (he, wi), cv2.FONT_HERSHEY_SIMPLEX, .30,
                        (255, 255, 255))

            # write each person's location to the output file
            stats.write(location + "\n")

        max = 0

        # if escape is hit exit
        if key in [27]:
            break


        #framerate (FPS) calculation
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = (framecount / elapsed_time)
        print(str(framecount) + " processed at " + str(fps) + "fps")

        #Packet Transmission timer
        elapsed_time_from_last_upload=end_time-last_upload_time

        # output a json packet to send to be picked up by the transmitter
        #currently set to transmit in 10 second intervals
        if (elapsed_time_from_last_upload >= 1):
            # dump to a json
            last_upload_time = time.time()
            outfile=open('/home/pi/Desktop/project_master/output_files/data.json','w')
            json.dump(detection, outfile)
            outfile.close()
            detection.clear()
            print("------------------PACKET WRITTEN FOR TRANSMISSION-----------------------")

        if (args[len(args) - 1]!="noshow"):
            # UI text show user how to quit, number of people and fps
            cv2.putText(frame, "Number of people:" + str(count), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.putText(frame, "Press ESC to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.putText(frame, "FPS: " + str(fps), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.putText(frame, "FRAME:" + str(framecount), (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

            # If noshow is not specified then display the detections to the screen
            cv2.imshow("Real Time Facial Recognition", frame)
        out.write(frame)
        framecount += 1

    cam.release()
    out.release()
    stats.close()
    cv2.destroyAllWindows()


detector(sys.argv)
