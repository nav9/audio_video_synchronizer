import math
import cv2
import mediapipe as mp
import time
from collections import deque
import plotly.express as px
import matplotlib.pyplot as plt

mediaPipeDraw = mp.solutions.drawing_utils
mediaPipeFaceMesh = mp.solutions.face_mesh

class Const:
    MILLISECONDS_IN_ONE_SECOND = 1000

class Landmark:
    def __init__(self, timestamp) -> None:
        self.timestamp = timestamp
        self.points = {} #{pointCode1: [x, y, z], pointCode2: [x, y, z]}
        self.lipSeparation = 0 #normalized distance the upper lip is from the lower lip (normalized, meaning the top of the head is considered 0% and tip of chin is considered 100%. Lip distances are normalized to between these two points before distances are calculated)

    def setPoint(self, pointCode, x, y, z):
        self.points[pointCode] = [x, y, z]

    def storeLipSeparation(self, normalizedAbsoluteDistance):
        self.lipSeparation = normalizedAbsoluteDistance

class VideoFaceProcessor:
    def __init__(self, videoSource, displayOutput=True) -> None:
        self.drawSettings = mediaPipeDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.videoSource = videoSource
        self.displayOutput = displayOutput
        self.minimumDetectionConfidence = 0.5
        self.minimumTrackingConfidence = 0.5
        self.topOfHead = 10
        self.tipOfChin = 152
        self.upperLipPoints = [82, 13, 312] #these are the canonical face model points that will be used to compare distance with the corresponding lower lip points
        self.lowerLipPoints = [87, 14, 317] #these points will be compared with the upper lip points (these point ID's are available here https://github.com/google/mediapipe/issues/1615)
        self.faces = {} #{faceID: deque[Landmark instance1, Landmark instance2, ...]}
        self.hardCodedFaceID = "face1" #TODO: face detection needs to be more generic before being able to assign a faceID to the self.faces dict

    def run(self):
        videoHandle = cv2.VideoCapture(self.videoSource)
        fps = videoHandle.get(cv2.CAP_PROP_FPS)
        print(f"Video has {fps}FPS. Processing...")
        #https://stackoverflow.com/questions/47743246/getting-timestamp-of-each-frame-in-a-video
        frameNumber = 0
        with mediaPipeFaceMesh.FaceMesh(min_detection_confidence=self.minimumDetectionConfidence, min_tracking_confidence=self.minimumTrackingConfidence) as detectedMesh:
            while videoHandle.isOpened():#as long as there are frames
                frameExists, theImage = videoHandle.read()
                if not frameExists:#reached end of video
                    break #for a stream, you'd use `continue` here
                #---preprocess
                theImage = cv2.cvtColor(theImage, cv2.COLOR_BGR2RGB)
                theImage.flags.writeable = False #a performance improvement (optional)
                processedImage = detectedMesh.process(theImage)
                #---Extract desired points
                theImage.flags.writeable = True
                theImage = cv2.cvtColor(theImage, cv2.COLOR_RGB2BGR)
                timestamp = videoHandle.get(cv2.CAP_PROP_POS_MSEC) / Const.MILLISECONDS_IN_ONE_SECOND
                print(f"Frame {frameNumber}, timestamp {timestamp}")
                if processedImage.multi_face_landmarks:
                    for detectedFace in processedImage.multi_face_landmarks:
                        if self.hardCodedFaceID not in self.faces:#face not present in dict
                            self.faces[self.hardCodedFaceID] = deque() #add new face
                        #mediaPipeDraw.draw_landmarks(image=theImage, landmark_list=land, connections=mediaPipeFaceMesh.FACEMESH_CONTOURS, landmark_drawing_spec=self.drawSettings, connection_drawing_spec=self.drawSettings)
                        pointIterator = 0
                        landmarkObject = Landmark(timestamp)
                        for pointOnFace in detectedFace.landmark:
                            if pointIterator in self.upperLipPoints or pointIterator in self.lowerLipPoints or pointIterator == self.topOfHead or pointIterator == self.tipOfChin:
                                landmarkObject.setPoint(pointIterator, pointOnFace.x, pointOnFace.y, pointOnFace.z)
                            pointIterator = pointIterator + 1  
                        self.faces[self.hardCodedFaceID].append(landmarkObject)
                frameNumber = frameNumber + 1
            videoHandle.release
            
        print(f"Finished processing. {len(self.faces[self.hardCodedFaceID])} landmark objects added for face {self.hardCodedFaceID}")

    def displayPoints(self):
        print("Detected points:")
        for faceID, landmarkDeque in self.faces.items():
            print(f"{len(landmarkDeque)} items are present in landmarkDeque")
            for landmark in landmarkDeque:
                points = []
                for pointCode, listOfXYZPoints in landmark.points.items():
                    points.append(f"{pointCode}:{listOfXYZPoints}")
                print(f"timestamp:{landmark.timestamp}: {points}") 

    def calculateLipMovement(self):
        for faceID, landmarkDeque in self.faces.items():
            print(f"{len(landmarkDeque)} landmarks")
            for landmark in landmarkDeque:
                faceHeight = abs(math.dist(landmark.points[self.topOfHead], landmark.points[self.tipOfChin]))
                #--calculate distances between opposing points on upper and lower lips
                totalDistance = math.dist(landmark.points[82], landmark.points[87])
                totalDistance = totalDistance + math.dist(landmark.points[13], landmark.points[14])
                totalDistance = totalDistance + math.dist(landmark.points[312], landmark.points[317])
                #---normalize distances based on face height
                normalizedLipSeparation = totalDistance * 100 / faceHeight #TODO: consider dividing totalDistance by the number of points on lip considered for comparison
                landmark.storeLipSeparation(normalizedLipSeparation)    
                print(f"{landmark.timestamp}: {landmark.lipSeparation}")

              