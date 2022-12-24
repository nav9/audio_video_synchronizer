from faceDetector import VideoFaceProcessor

if __name__ == '__main__':
    videoSource = "thePause.mp4"
    faceProcessor = VideoFaceProcessor(videoSource)
    faceProcessor.run()
    #faceProcessor.displayPoints()
    