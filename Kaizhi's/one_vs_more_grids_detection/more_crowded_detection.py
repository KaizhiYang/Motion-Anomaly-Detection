import cv2
import numpy as np

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_fps = 24  # Output frames per second
output_vid = "CrowdTracking/Kaizhi's/one_vs_more_grids_detection/output_4_grids.mp4"
out = cv2.VideoWriter(output_vid, fourcc, output_fps, (608, 1080))  # Change dimensions as needed

def generate_grid_coordinates(n, video_width, video_height):
    # Calculate the grid size
    grid_width = video_width // n
    grid_height = video_height // n
    grid_size = (grid_width, grid_height)

    # Initialize a list to store grid coordinates and sizes
    grid_list = []

    # Generate grid coordinates and sizes
    for i in range(n):
        for j in range(n):
            x_start = i * grid_width
            y_start = j * grid_height
            x_end = x_start + grid_width
            y_end = y_start + grid_height
            grid_list.append(((x_start, y_start), (x_end, y_end)))

    return grid_size, grid_list

input_vid = "CrowdTracking/Kaizhi's/crowd_people_video/crowded_people.mp4"
input_cam = 2

# Define number of grid and grid size
num_grid = 4
num_grid = int(np.sqrt(num_grid))
grid_size, _ = generate_grid_coordinates(num_grid, 608, 1080)  

## set up input type and classes
cap = cv2.VideoCapture(input_vid) # input could be video or camera 2
file = open("CrowdTracking/Kaizhi's/classes.txt","r")
classes = file.read().split('\n')
print(classes)

## random colors for bounding boxes
numClass = len(classes)
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(numClass, 3), dtype='uint8')

## read network model
net = cv2.dnn.readNetFromONNX("CrowdTracking/Kaizhi's/yolov5x.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

## detect each img of video
while True:
    # read img and process on it
    ret, img = cap.read()
    if ret is False:
        break
    
    # Calculate the number of rows and columns of grids/windows
    frame_height, frame_width, _ = img.shape
    num_rows = frame_height // grid_size[1]
    num_cols = frame_width // grid_size[0]

    person_count = 0

    # Iterate over the frame using the grid/window
    for y in range(num_rows):
        for x in range(num_cols):
            # Calculate the coordinates of the grid/window
            y_start = y * grid_size[1]
            x_start = x * grid_size[0]
            y_end = y_start + grid_size[1]
            x_end = x_start + grid_size[0]

            # Draw a rectangle on the original image to represent the grid
            cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

            # Extract the grid/window
            grid = img[y_start:y_end, x_start:x_end]

            # generate blob and load it to get detections
            blob = cv2.dnn.blobFromImage(grid, scalefactor=1/255, size=(640, 640), mean=[0, 0, 0], swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward()[0]  # cx, cy , w, h, confidence, 80 class_scores in detections
  
            # class_ids, confidences, boxes --need to find
            classes_ids = []
            confidences = []
            boxes = []
            
            rows = detections.shape[0] # each detection

            # data used to restore bounding boxes to original size for img
            img_width, img_height = grid.shape[1], grid.shape[0]
            x_scale = img_width/640
            y_scale = img_height/640

            # collect objects' info
            for i in range(rows):
                row = detections[i]
                confidence = row[4]
                if confidence > 0.5:
                    classes_score = row[5:]
                    ind = np.argmax(classes_score)
                    if classes_score[ind] > 0.5:
                        classes_ids.append(ind)
                        confidences.append(confidence)
                        
                        # covert center x, center y, width, and height to top left and bottom right coordinate
                        cx, cy, w, h = row[:4]
                        x1 = int((cx- w/2)*x_scale)
                        y1 = int((cy-h/2)*y_scale)
                        width = int(w * x_scale)
                        height = int(h * y_scale)
                        
                        box = np.array([x1,y1,width,height])
                        boxes.append(box)

            # remove the duplicate and overlapping bounding boxes
            indices = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.5)

            # show boxes and info
            for i in indices:
                x1,y1,w,h = boxes[i]
                label = classes[classes_ids[i]]
                conf = confidences[i]
                text = label + "{:.2f}".format(conf)
                color = [int(c) for c in COLORS[classes_ids[i]]]
                if label == "person":
                    person_count += 1
                    cv2.rectangle(grid,(x1,y1),(x1+w,y1+h),color,1)
                    cv2.putText(grid, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,color,1)

    out.write(img)  # Write the frame to the output video
    print("Number of person detected (each frame): " + str(person_count) + "\n")

    cv2.imshow("person_VIDEO",img)
    k = cv2.waitKey(500) # 5 seconds = 5 * 1000
    if k == ord('q'):
        break

# Release the VideoWriter
out.release()

cap.release()
cv2.destroyAllWindows()