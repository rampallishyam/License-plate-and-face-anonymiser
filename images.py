"""
Implement face and license plate masking using YOLO from ultralytics
YOLO has three modes and it can perform three tasks
the Modes of YOLO are train, val and predict
The tasks that YOLO can perform are detect, segment and classify
In this script we are going to use detect task

"""

from ultralytics import YOLO
import os

# Use the both lines to check the status of ultralytics and 
# the GPUs that it has access to depending on how you install torch
# import ultralytics
# ultralytics.checks()

# Create a model from scratch. 
# More tutorials on getting started on using YOLO models here: https://docs.ultralytics.com/usage/python/ 
model = YOLO('yolov8x.yaml')
# Import the pretrained yolov8n model
model = YOLO('yolov8x.pt')

# Train the model using the 'data.yaml' dataset for 3 epochs
results = model.train(data=os.getcwd()+'\data.yaml', epochs=3)

# Evaluate the model's performance on the validation set
results = model.val(data = os.getcwd()+'\data.yaml')

# Perform object detection on an image using the model
results = model(os.getcwd()+"\sampleimagefortesting.jpg")



