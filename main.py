from faceDetector import VideoFaceProcessor

if __name__ == '__main__':
    videoSource = "vid.mp4"
    faceProcessor = VideoFaceProcessor(videoSource)
    faceProcessor.run()
    #faceProcessor.displayPoints()
    