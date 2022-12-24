import os
import psutil
import shlex #useful for recognizing quotes inside a command to be split
import subprocess
from faceDetector import VideoFaceProcessor

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
    videoSource = "thePause.mp4"
    filenameWithoutExtension = os.path.splitext(videoSource)[0] #remove extension
    command = f"ffmpeg -y -i {videoSource} -vn -ac 1 {filenameWithoutExtension}.wav"    
    runCommand(command)
    #faceProcessor = VideoFaceProcessor(videoSource)
    #faceProcessor.run()
    #faceProcessor.displayPoints()

    