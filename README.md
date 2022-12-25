# What this project does  
Synchronises audio and video based on lip movement of speakers. This is a simplistic implementation that is far from being generic.
It detects phases where speech has started after phases of silence, and at such points in the video, it looks for matching points in the audio timeline nearby. The first such audio-video offset found, is used to correct the audio-video async.  
MediaPipe is used for detecting the points on the face and lips.
  
# Running 
Install the necessary Python packages and simply use `python3 main.py`.  
  
# Install requirements  
TODO.
  
# Attribution  
* The Pause Video: https://www.youtube.com/watch?v=7l1Tom9q8Ic
* Voice activity detection: https://github.com/wiseman/py-webrtcvad
