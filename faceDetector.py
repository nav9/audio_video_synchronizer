import math
import cv2
import mediapipe as mp
import time
from collections import deque

mediaPipeDraw = mp.solutions.drawing_utils
mediaPipeFaceMesh = mp.solutions.face_mesh

class Const:
    MILLISECONDS_IN_ONE_SECOND = 1000

class Landmark:
    def __init__(self, timestamp) -> None:
        self.timestamp = timestamp
        self.points = {} #{pointCode1: [x, y, z], pointCode2: [x, y, z]}
        self.lipSeparation = 0 #normalized distance the upper lip is from the lower lip (normalized, meaning the top of the head is considered 0% and tip of chin is considered 100%. Lip distances are normalized to between these two points before distances are calculated)
        self.speaking = True

    def setPoint(self, pointCode, x, y, z):
        self.points[pointCode] = [x, y, z]

    def storeLipSeparation(self, normalizedAbsoluteDistance):
        self.lipSeparation = normalizedAbsoluteDistance

class VideoFaceProcessor:
    def __init__(self, videoSource, displayMesh=False) -> None:
        self.drawSettings = mediaPipeDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.videoSource = videoSource
        self.displayMesh = displayMesh
        self.minimumDetectionConfidence = 0.5
        self.minimumTrackingConfidence = 0.5
        self.topOfHead = 10
        self.tipOfChin = 152
        self.upperLipPoints = [82, 13, 312] #these are the canonical face model points that will be used to compare distance with the corresponding lower lip points
        self.lowerLipPoints = [87, 14, 317] #these points will be compared with the upper lip points (these point ID's are available here https://github.com/google/mediapipe/issues/1615)
        self.faces = {} #{faceID: deque[Landmark instance1, Landmark instance2, ...]}
        self.hardCodedFaceID = "face1" #TODO: face detection needs to be more generic before being able to assign a faceID to the self.faces dict
        self.fps = None
        self.fontDisplayPosition = (10, 30)
        self.fontScale = 0.5
        self.fontColor = (0, 255, 0)
        self.fontThickness = 2      
        self.pauseDuration = None

    def run(self):
        videoHandle = cv2.VideoCapture(self.videoSource)
        self.fps = videoHandle.get(cv2.CAP_PROP_FPS)
        self.pauseDuration = int((7/30) * self.fps) #because during an experiment it was seen that in a 30FPS video, if 7 consecutive frames were with mouth shut, it could be a pause
        print(f"Video has {self.fps}FPS and pause detection duration = {self.pauseDuration} frames Processing...")
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
                if self.displayMesh:
                    self.displayVideo(theImage, f'FPS: {int(self.fps)}')
                timestamp = videoHandle.get(cv2.CAP_PROP_POS_MSEC) / Const.MILLISECONDS_IN_ONE_SECOND
                print(f"Frame {frameNumber}, timestamp {timestamp}")
                if processedImage.multi_face_landmarks:
                    for detectedFace in processedImage.multi_face_landmarks:
                        if self.hardCodedFaceID not in self.faces:#face not present in dict
                            self.faces[self.hardCodedFaceID] = deque() #add new face
                        if self.displayMesh:
                            mediaPipeDraw.draw_landmarks(image=theImage, landmark_list=detectedFace, connections=mediaPipeFaceMesh.FACEMESH_CONTOURS, landmark_drawing_spec=self.drawSettings, connection_drawing_spec=self.drawSettings)
                        pointIterator = 0
                        landmarkObject = Landmark(timestamp)
                        for pointOnFace in detectedFace.landmark:
                            if pointIterator in self.upperLipPoints or pointIterator in self.lowerLipPoints or pointIterator == self.topOfHead or pointIterator == self.tipOfChin:
                                landmarkObject.setPoint(pointIterator, pointOnFace.x, pointOnFace.y, pointOnFace.z)
                            pointIterator = pointIterator + 1  
                        self.faces[self.hardCodedFaceID].append(landmarkObject)
                frameNumber = frameNumber + 1                
            videoHandle.release            
        #print(f"Finished processing. {len(self.faces[self.hardCodedFaceID])} landmark objects added for face {self.hardCodedFaceID}")
        self.calculateLipMovement()
        
    def displayVideo(self, image, textToDisplay):        
        cv2.putText(image, textToDisplay, self.fontDisplayPosition, cv2.FONT_HERSHEY_DUPLEX, self.fontScale, self.fontColor, self.fontThickness)
        cv2.imshow('Lip movement detection', image)
        cv2.waitKey(1)

    def displayPoints(self):
        print("Detected points:")
        for faceID, landmarkDeque in self.faces.items():
            #print(f"{len(landmarkDeque)} items are present in landmarkDeque")
            for landmark in landmarkDeque:
                points = []
                for pointCode, listOfXYZPoints in landmark.points.items():
                    points.append(f"{pointCode}:{listOfXYZPoints}")
                #print(f"timestamp:{landmark.timestamp}: {points}") 

    def calculateLipMovement(self):        
        #---calculate average lip separation distance for all frames
        for faceID, landmarkDeque in self.faces.items():
            #print(f"{len(landmarkDeque)} landmarks")
            prevLipSeparation = 0
            for landmark in landmarkDeque:#landmark is a landmarkObject
                faceHeight = abs(math.dist(landmark.points[self.topOfHead], landmark.points[self.tipOfChin]))
                #--calculate distances between opposing points on upper and lower lips
                averageDistance = self.calculateAverageLipOpenDistance(landmark.points)
                #---normalize distances based on face height
                normalizedLipSeparation = averageDistance * 100 / faceHeight 
                landmark.storeLipSeparation(normalizedLipSeparation)                    
                t = float("{:.2f}".format(landmark.timestamp)); d = float("{:.2f}".format(abs(landmark.lipSeparation - prevLipSeparation))); s = float("{:.2f}".format(landmark.lipSeparation));
                print(f"Time{t} diff:{d} Sep:{s}")
                prevLipSeparation = landmark.lipSeparation
            self.determineSilencePhases(landmarkDeque)
    
    def getDetectedSilences(self):
        return self.faces[self.hardCodedFaceID] #returns the deque of landmark objects
        
    def determineSilencePhases(self, landmarkDeque):
        #---use a sliding window to determine if the person is speaking (assuming 4 syllables per second https://en.wikipedia.org/wiki/Speech_tempo)
        pastFew = deque(maxlen=self.pauseDuration)
        mouthOpeningThreshold = 1 #distance of lip separation        
        for i in range(0, len(landmarkDeque)):
            pastFew.append(landmarkDeque[i].lipSeparation)
            if all(i < mouthOpeningThreshold for i in pastFew):#mouth has been closed for a few frames
                for j in range(i, i-len(pastFew), -1):#go in reverse and mark those landmark objects as mouth closed 
                    landmarkDeque[j].speaking = False
        #self.showDetectedSilencePhases(landmarkDeque)

    def calculateAverageLipOpenDistance(self, points):
        totalDistance = 0
        for i in range(len(self.upperLipPoints)):
            totalDistance += math.dist(points[self.upperLipPoints[i]], points[self.lowerLipPoints[i]])
        return totalDistance / len(self.upperLipPoints)

    def showDetectedSilencePhases(self, landmarkDeque):
        for landmark in landmarkDeque:
            print(f"Time:{landmark.timestamp} speaking:{landmark.speaking}")        
        videoHandle = cv2.VideoCapture(self.videoSource)
        frameNumber = 0
        while videoHandle.isOpened():#as long as there are frames
            frameExists, theImage = videoHandle.read()
            if not frameExists:#reached end of video
                break #for a stream, you'd use `continue` here
            #---preprocess
            theImage.flags.writeable = False #a performance improvement (optional) 
            textToDisplay = "?"
            if landmarkDeque[frameNumber].speaking:
                textToDisplay = "Speaking"
            if not landmarkDeque[frameNumber].speaking:
                textToDisplay = "Silence"            
            self.displayVideo(theImage, textToDisplay)
            time.sleep(1/self.fps)
            frameNumber = frameNumber + 1