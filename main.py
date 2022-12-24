import os
import psutil
import shlex #useful for recognizing quotes inside a command to be split
import subprocess
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
    videoSource = "thePause.mp4"
    nonSpeechFilterLevel = 2
    filenameWithoutExtension = os.path.splitext(videoSource)[0] #remove extension
    #---Create an audio file from the video file
    command = f"ffmpeg -y -i {videoSource} -vn -ac 1 {filenameWithoutExtension}{waveExtension}"    
    #runCommand(command)
    #---Analyze audio to detect speech
    vad = VoiceActivityDetector()
    vad.run(f"{filenameWithoutExtension}{waveExtension}", nonSpeechFilterLevel)
    audioMarkers = vad.getSpeechDetectedSections()
    #---Analyze lip movements to detect silences
    #faceProcessor = VideoFaceProcessor(videoSource)
    #faceProcessor.run()
    #videoMarkers = faceProcessor.getDetectedSilences()
    ##faceProcessor.displayPoints()
    #---Check for Audio Video sync issues
    print(f"Num. audio samples {len(audioMarkers)}") 
    #print(f"Num. video samples {len(videoMarkers)}")


    