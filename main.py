import os
import psutil
import shlex #useful for recognizing quotes inside a command to be split
import subprocess
from collections import deque
from faceDetector import VideoFaceProcessor
from voiceActivityDetection import VoiceActivityDetector

def runCommand(command):
    command = shlex.split(command)       
    try:
        process = subprocess.Popen(command)        
        #---keep polling
        while True: 
            returnCode = process.poll() #checking if process ended (could also use psutil to check)
            if returnCode == None: #process still running
                pass
            else: #process completed
                break
    except subprocess.CalledProcessError as e:
        print("Ran into some errors:", e) 

if __name__ == '__main__':
    waveExtension = ".wav"
    #videoSource = "thePause.mp4"
    videoSource = "thePause_withAudioOffset.mp4"
    nonSpeechFilterLevel = 1
    filenameWithoutExtension = os.path.splitext(videoSource)[0] #remove extension
    #---Create an audio file from the video file
    command = f"ffmpeg -y -i {videoSource} -vn -ac 1 {filenameWithoutExtension}{waveExtension}"    
    runCommand(command)
    #---Analyze audio to detect speech
    print("Processing audio")
    vad = VoiceActivityDetector()
    vad.run(f"{filenameWithoutExtension}{waveExtension}", nonSpeechFilterLevel)
    audioMarkers = vad.getSpeechDetectedSections()
    #---Analyze lip movements to detect silences
    print("Processing video")
    faceProcessor = VideoFaceProcessor(videoSource)
    faceProcessor.run()
    videoMarkers = faceProcessor.getDetectedSilences()
    ##faceProcessor.displayPoints()
    #---Check for Audio Video sync issues
    print(f"Num. audio points {len(audioMarkers)}") 
    print(f"Num. video points {len(videoMarkers)}")
    numFramesToCheck = 15
    detectionWindow = deque(maxlen=numFramesToCheck)
    videoCrucialPoints = []; audioCrucialPoints = []    
    print('--------- video')
    for frame in videoMarkers:        
        #---detect the point at which speaking starts after a pause
        if len(detectionWindow) == numFramesToCheck and all(i == False for i in detectionWindow) and frame.speaking == True:#the False means notSpeaking and True means speaking
            print("Speaking started after a silence at: ", frame.timestamp, frame.speaking)
            videoCrucialPoints.append(frame.timestamp)
        detectionWindow.append(frame.speaking)
    print('--------- audio')
    detectionWindow = deque(maxlen=numFramesToCheck)
    for frame in audioMarkers:        
        #---detect the point at which speaking starts after a pause
        if len(detectionWindow) == numFramesToCheck and all(i == False for i in detectionWindow) and frame.speaking == True:#the False means notSpeaking and True means speaking
            print("Speaking started after a silence at: ", frame.timestamp, frame.speaking)
            audioCrucialPoints.append(frame.timestamp)
        detectionWindow.append(frame.speaking)
    for i in range(0, len(videoCrucialPoints)):
        diff = videoCrucialPoints[i] - audioCrucialPoints[i]
        if diff == 0:
            print(f"point {i}: Audio and video are in sync")
        if diff > 0:
            print(f"point {i}: Video is ahead of audio by {diff}s")
        if diff < 0:
            print(f"point {i}: Audio is ahead of video by {diff}s")            