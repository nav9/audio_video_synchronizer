import cv2
import mediapipe as mp
import time
from collections import deque
import plotly.express as px
import matplotlib.pyplot as plt

mediaPipeDraw = mp.solutions.drawing_utils
mediaPipeFaceMesh = mp.solutions.face_mesh

class Landmark:
  def __init__(self, timestamp) -> None:
    self.timestamp = timestamp
    self.points = {} #{pointCode1: [x, y, z], pointCode2: [x, y, z]}

  def setPoint(self, pointCode, x, y, z):
    self.points[pointCode] = [x, y, z]

class VideoFaceProcessor:
  def __init__(self, videoSource, displayOutput=True) -> None:
    self.drawSettings = mediaPipeDraw.DrawingSpec(thickness=1, circle_radius=1)
    self.videoSource = videoSource
    self.displayOutput = displayOutput
    self.minimumDetectionConfidence = 0.5
    self.minimumTrackingConfidence = 0.5
    self.upperLipPoints = [82, 13, 312] #these are the canonical face model points that will be used to compare distance with the corresponding lower lip points
    self.lowerLipPoints = [87, 14, 317] #these points will be compared with the upper lip points (these point ID's are available here https://github.com/google/mediapipe/issues/1615)
    self.faces = {} #{faceID: deque[Landmark instance1, Landmark instance2, ...]}
    self.hardCodedFaceID = "face1" #TODO: face detection needs to be more generic before being able to assign a faceID to the self.faces dict

  def run(self):
    videoHandle = cv2.VideoCapture(self.videoSource)
    print("Processing...")
    startTime = time.time()
    with mediaPipeFaceMesh.FaceMesh(min_detection_confidence=self.minimumDetectionConfidence, min_tracking_confidence=self.minimumTrackingConfidence) as detectedMesh:
      while videoHandle.isOpened():
        success, theImage = videoHandle.read()
        if not success:#reached end of video
          break #for a stream, you'd use `continue` here
        #---preprocess
        theImage = cv2.cvtColor(cv2.flip(theImage, 1), cv2.COLOR_BGR2RGB)
        theImage.flags.writeable = False #a performance improvement (optional)
        processedImage = detectedMesh.process(theImage)
        #---Extract desired points
        theImage.flags.writeable = True
        theImage = cv2.cvtColor(theImage, cv2.COLOR_RGB2BGR)
        timestamp = time.time() - startTime #unit is seconds
        if processedImage.multi_face_landmarks:
          for detectedFace in processedImage.multi_face_landmarks:
            if self.hardCodedFaceID not in self.faces:#face not present in dict
              self.faces[self.hardCodedFaceID] = deque() #add new face
            #mediaPipeDraw.draw_landmarks(image=theImage, landmark_list=land, connections=mediaPipeFaceMesh.FACEMESH_CONTOURS, landmark_drawing_spec=self.drawSettings, connection_drawing_spec=self.drawSettings)
            pointIterator = 0
            landmarkObject = Landmark(timestamp)
            for pointOnFace in detectedFace.landmark:
              if pointIterator in self.upperLipPoints or pointIterator in self.lowerLipPoints:
                landmarkObject.setPoint(pointIterator, pointOnFace.x, pointOnFace.y, pointOnFace.z)
              pointIterator = pointIterator + 1    
          self.faces[self.hardCodedFaceID].append(landmarkObject)
    videoHandle.release
    print("Finished processing")

  def displayPoints(self):
    for faceID, landmarkDeque in self.faces.items():
      for landmark in landmarkDeque:
        points = []
        for pointCode, listOfXYZPoints in landmark.points.items():
          points.append(f"{pointCode}:{listOfXYZPoints}")
        print(f"timestamp:{landmark.timestamp}: {points}") 
