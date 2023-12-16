import cv2
from ultralytics import YOLO
import argparse

# arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_yolo', default='model/model.pt', help='path for the Yolo model')
parser.add_argument('--source_video_path', default='videos/1.mp4', help=r'path of the input video file name with extension, for ex: C:\Users\rampa\Desktop\video.mp4')
parser.add_argument('--output_video_path', default='output_videos/blurred_video.mp4',help='path of the output video file name with extension')
args = parser.parse_args()

# Load a pretrained YOLOv8n model
model = YOLO(args.model_yolo)

# Define path to video file
video_path = args.source_video_path

# Load video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the output video file
output_path = args.output_video_path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through each frame in the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model.predict(source=frame, show=False)

    if sum(results[0].names.keys()) > 0:
        bboxes = results[0].boxes.data[:,:4].cpu().numpy() 
    
    for bbox in bboxes:
        x, y, x2, y2 = map(int, bbox)
        roi = frame[y:y2, x:x2]
        roi = cv2.GaussianBlur(roi, (99, 99), 0)  # You can adjust the kernel size for more or less blur
        frame[y:y2, x:x2] = roi

    # Write the frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# print(results)