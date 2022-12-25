import os
import shlex #useful for recognizing quotes inside a command to be split
import subprocess
from collections import deque
import matplotlib.pyplot as plt
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

def plotSpeechDetected(audioMarkers, videoMarkers):
    audioTimestamps = []; videoTimestamps = []
    audioSpeaking = []; videoSpeaking = []
    for frame in videoMarkers:        
        videoSpeaking.append(frame.speaking)
        videoTimestamps.append(frame.timestamp) 
    for frame in audioMarkers:        
        audioSpeaking.append(frame.speaking)
        audioTimestamps.append(frame.timestamp)         
    fig, axs = plt.subplots(2)
    fig.suptitle('Speech detected (crests)')
    axs[0].plot(audioTimestamps, audioSpeaking); axs[0].set(xlabel='timestamp', ylabel='audio')
    axs[1].plot(videoTimestamps, videoSpeaking); axs[1].set(xlabel='timestamp', ylabel='video')
    plt.show() 

def searchNearbyAudioTimestamps(audioCrucialPoints, videoTimestamp):
    maxOffset = 1.5 #audio video may be out of sync by a max of these many seconds
    nearestAudioIndex = None #audio index of speech that's closest to the video timestamp
    closestValue = 999
    for i in range(0, len(audioCrucialPoints)):
        val = abs(videoTimestamp - audioCrucialPoints[i])
        if val <= maxOffset:#is within range of the video timestamp offset allowed
            if val < closestValue:#find the closest offset detected
                closestValue = val
                nearestAudioIndex = i
    return nearestAudioIndex


if __name__ == '__main__':
    waveExtension = ".wav"
    mp4Extension = ".mp4"
    #videoSource = "thePause2.mp4"
    videoSource = "thePause2_withAudioOffset.mp4"
    nonSpeechFilterLevel = 1
    filenameWithoutExtension = os.path.splitext(videoSource)[0] #remove extension
    
    #---Create an audio file from the video file
    command = f"ffmpeg -hide_banner -loglevel error -y -i {videoSource} -vn -ac 1 {filenameWithoutExtension}{waveExtension}"    
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
    plotSpeechDetected(audioMarkers, videoMarkers)
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
    allDiffs = []
    for i in range(0, len(videoCrucialPoints)):
        offsetIndex = searchNearbyAudioTimestamps(audioCrucialPoints, videoCrucialPoints[i])
        if offsetIndex == None:
            print(f"No nearby audio found for point {i} of video")
        diff = videoCrucialPoints[i] - audioCrucialPoints[offsetIndex]
        if diff == 0:
            print(f"point {i}: Audio and video are in sync")
        if diff > 0:
            print(f"point {i}: Video is ahead of audio by {diff}s")
        if diff < 0:
            print(f"point {i}: Audio is ahead of video by {diff}s")            
        allDiffs.append(diff)
    
    #---calculate an approximate offset to perform
    offset = allDiffs[0] #most reliable async
    print(f"Offset is {offset}s. Adjusting audio in video by this much.")
    
    #---Do the sync
    command = f"ffmpeg -hide_banner -loglevel error -y -i {videoSource} -itsoffset {offset} -i {videoSource} -map 0:v -map 1:a -c copy sync{mp4Extension}"  #https://superuser.com/questions/982342/in-ffmpeg-how-to-delay-only-the-audio-of-a-mp4-video-without-converting-the-au
    runCommand(command)     
    print(f"\n\nProgram complete. Synched file is: sync{mp4Extension}")            