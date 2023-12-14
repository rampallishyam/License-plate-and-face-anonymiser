import cv2
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('model/model.pt')

# Define path to video file
video_path = 'videos/4.mp4'

# Load video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the output video file
output_path = 'output_videos/blurred_video4.mp4'
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