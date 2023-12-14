"""
This script will take video as input, use a pre-trained model
and outputs video with detections

"""

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('model/model.pt')

# Define path to video file
video = 'videos/2.mp4'

# Define path to image file
source_img = 'sampleimagefortesting.jpg'

# Run inference on the source
results = model.predict(source = 0, show=True) 
# print(results)