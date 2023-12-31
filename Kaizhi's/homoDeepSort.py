import cv2
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# DeepSORT -> Importing DeepSORT.
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

# choose input source for detection
input_vid = "CrowdTracking/Kaizhi's/people.mp4"
input_cam = 2

# video saving
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vidFile = "CrowdTracking/Kaizhi's/test1.mp4"
vw = cv2.VideoWriter(vidFile, fourcc, 24, (2302,1302),1) # resolution has to align with the image you want to save as a video
videoSaver = True  # 1: save video  0: not save

# data saver
# Specify the file path (change it to your desired file path)
PATH = "CrowdTracking/Kaizhi's/data.txt"

# data for Homography process
top_left = (41,4)
top_right = (1794,5)
bot_left = (0,0)
bot_right = (0,0)
srcCoordinates = np.array([top_right, top_left, bot_left, bot_right], np.int32)

# data for invade detection
top_left_frbd = (600,385)
top_right_frbd = (900,380)
bot_left_frbd = (455,870)
bot_right_frbd = (890,870)
forbidden_area = np.array([top_right_frbd, top_left_frbd, bot_left_frbd, bot_right_frbd], np.int32)

# Threshold for detecting anomalies
anomly_threshold = 0.001

# get the m and n in the equation y = mx + n
def getMN(pointA, pointB):
    Xa, Ya = pointA
    Xb, Yb = pointB
    factor_matrix = np.array(([Xa,1],[Xb,1]))
    y_matrix = np.array(([Ya, Yb]))
    m, n = np.linalg.solve(factor_matrix, y_matrix)
    return m,n

def getOutRangeXY(m, n, y): 
    newX = (y - n)/m
    return (newX, y)

def getTransformed(img, srcPts, flag):
    (tl, tr, br, bl) =  srcPts
    
    width1 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width2 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    maxWidth = int(max(width1, width2))
    
    height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = int(max(height1, height2))
    
    dst = np.array([[0,0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]], np.float32)
    
    hMatrix = cv2.getPerspectiveTransform(srcPts, dst)
    
    if flag:
        result = cv2.warpPerspective(img, hMatrix, (maxWidth, maxHeight))
    else:
        result = hMatrix
    return result

def getCoordinate(x,y,w,h):
    new_x = x+w/2
    new_y = y+h
    return np.array([[new_x], [new_y], [1]], np.float32)

def threeImgShow(img1, img2, img3, videoSaver):
    # Resize images to the same dimensions (optional)
    width, height = 1151, 651
    img1 = cv2.resize(img1, (width, height))
    img2 = cv2.resize(img2, (width, height))
    img3 = cv2.resize(img3, (width, height))
    
    # Determine the canvas size to accommodate all images
    canvas_width = width * 2
    canvas_height = height * 2
    
    # Create a blank canvas with white background
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # Paste the images onto the canvas
    canvas[0:height, 0:width] = img1  # Top left
    canvas[0:height, width:width*2] = img2  # Top right
    canvas[height:height*2, 0:width] = img3  # Bottom left   
    
    # Save the image as video
    if videoSaver == True:
        vw.write(canvas)
    
    # Display the final canvas
    cv2.imshow('Combined Images', canvas)
    
    
def forbiddenAreaTrans(frbd_area, h):
    result = []
    for xy in frbd_area:
        list1 = list(xy)
        list1.append(1)
        trasformed_xy = np.dot(h, list1)
        newX, newY, temp = trasformed_xy/trasformed_xy[2]
        result.append((int(newX), int(newY)))
    return np.array(result)

# invade detection method: mask image composed of 1 and 0 with 1 as walkable and 0 as not walkable
def maskImgDetect(mask_img, people):
    person_x, person_y = people
    mask_width, mask_height = mask_img.shape[1], mask_img.shape[0]
    person_x = int(max(0, min(person_x, mask_width-1)))
    person_y = int(max(0, min(person_y, mask_height-1)))
    if mask_img[person_y, person_x] == 0:
        return True
    else:
        return False
    
# covert x and y coordinates system into magnitude and angle system
def coordinateToPolar(x_p, y_p):
    magnitude = np.sqrt((x_p) ** 2 + (y_p) ** 2)
    theta = np.arctan2(y_p, x_p)
    degree = np.rad2deg(theta)
    return magnitude, degree

# Use the extend method to combine the lists
def combineLists(lists):
    combinedList = []
    for sublist in lists:
        combinedList.extend(sublist)
    return combinedList

# Normal Distribution Plot
def normalDistGraph(mean, std_dev, src_name):

    # Create an array of x values
    x = np.linspace(-10*mean, 10*mean, 100)  # Choose an appropriate range
    if src_name == "Angle":
        x = np.linspace(-8*std_dev, 8*std_dev, 100)

    # Calculate the PDF for each x value
    pdf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-((x - mean)**2) / (2 * std_dev**2))

    # fromatting numbers
    formatted_mean = "{:.4f}".format(mean)
    formatted_std_dev = "{:.4f}".format(std_dev)

    # Plot the normal distribution
    plt.plot(x, pdf, label='Normal Distribution')
    plt.xlabel("X (" + src_name + ")")
    plt.ylabel('Probability Density')
    plt.legend()
    plt.title(f'Normal Distribution (μ={formatted_mean}, σ={formatted_std_dev})')
    plt.grid(True)
    

# Plot scatter graph
def scatterGraph(total_mag, total_ang):
    # Create the scatter plot
    plt.scatter(total_mag, total_ang, color='blue', marker='o')

    # Optional: Add labels and a title
    plt.xlabel('Magnitude')
    plt.ylabel('Angle')
    plt.title('Scatter Plot')

    # Optional: Add a legend
    plt.legend()

# Create KDE (Kernel Density Estimate) plot
def KDEgraph(total_mag, total_ang):

    # Create a 2D KDE plot
    sns.kdeplot(x=total_mag, y=total_ang, cmap="bwr", fill=True, levels=8, cbar=True)

    # Limit x axis from 0 to 20
    plt.xlim(0,20)

    # Add labels and a title
    plt.xlabel("Magnitude")
    plt.ylabel("Angle")
    plt.title("2D KDE Plot")

# Plot graphs: two normal distributions + one scatter graph
def plotGraphs(total_mag, total_ang, mean_mag, std_dev_mag, mean_ang, std_dev_ang):
    plt.subplot(2,2,1)
    scatterGraph(total_mag, total_ang)
    plt.subplot(2,2,3)
    KDEgraph(total_mag, total_ang)
    plt.subplot(2,2,2)
    normalDistGraph(mean_mag, std_dev_mag, "Magnitude")
    plt.subplot(2,2,4)
    normalDistGraph(mean_ang, std_dev_ang, "Angle")
    plt.tight_layout()
    plt.show()

def getKdeValue(x_values, y_values):
    # Fit a 2D KDE estimator to the data
    kde = gaussian_kde([x_values, y_values])

    # Evaluate the KDE estimator at specific points to get density values
    points_to_evaluate = np.vstack([x_values, y_values])
    density_values = kde(points_to_evaluate)
    return density_values

def dataSaver(data1, data2, data3, file_path):
    # Open the file in write mode ('w')
    with open(file_path, 'w') as file:
        # Write data to the file
        for i in range(len(data1)):
            file.write(f"{data1[i]}\t{data2[i]}\t{data3[i]}\n")
        min_data3 = min(data3)
        max_data3 = max(data3)
        file.write(f"{min_data3}\t{max_data3}\n")
    # File is automatically closed when the 'with' block exits

    print(f"Data has been saved to {file_path}\n")

def dataGetter(file_path):
    data = []
    min_max = (-1,-1)
    
    # Open the file in read mode ('r')
    with open(file_path, 'r') as file:
        # Read each line of the file
        for line in file:
            # Split the line into two values (assuming tab-separated values)
            values = line.strip().split('\t')
            if len(values) == 3:
                # Convert the values to float and append to respective lists
                value1 = float(values[0])
                value2 = float(values[1])
                value3 = float(values[2])
                data.append([value1, value2, value3])
            elif len(values) == 2:
                value1 = float(values[0])
                value2 = float(values[1])
                min_max = [value1, value2]
    print(f"Data has been saved to local variables")
    print("Min and max density: ", min_max)
    print("\n")
    return data, min_max

# Find density value by Magnitude and Angle
def findDensityByMA(m, a, data):
    density = -1
    for i in range(len(data)):
        if np.isclose(data[i][0], m, atol=1e-8) and np.isclose(data[i][1], a, atol=1e-8):
            density = data[i][2]
    return density

# Normalize value in range of min and max, then assign color to that value based on normalized value
def value_to_color(value, min_value, max_value, colorGradients):
    
    # Calculate the normalized value within the range [0, 1]
    normalized_value = (value - min_value) / (max_value - min_value)
    index = int(max(normalized_value * 512 - 2, 0))
    
    # Interpolate between blue and red based on the normalized value
    interpolated_color = colorGradients[index]

    # Return the interpolated color as an integer tuple (B, G, R)
    return interpolated_color

# return array of array which contains BGR from Blue to Red
def colorBlue2Red():
    # Define the number of steps from Blue to Red GBR
    num_steps = 511 

    # Create a gradient from blue to red
    gradient = np.zeros((num_steps, 3))
    count = 255

    # Store colors in gradient
    for i in range(num_steps):
        if (i < 255):
            gradient[i][0] = i
            gradient[i][1] = i
            gradient[i][2] = 255
        else:
            gradient[i][0] = 255
            gradient[i][1] = count
            gradient[i][2] = count
            count -= 1
    return gradient


if __name__ == '__main__':
    # DeepSORT -> Intializing tracker.
    max_cosine_distance = 0.4
    nn_budget = None
    model_filename = "CrowdTracking/Kaizhi's/mars-small128.pb"
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    
    fileChecker = 1
    magnitude_angle_density = -1
    min_max_density = [-1,-1]
    try:
    # Attempt to open a file that may not exist
        with open(PATH, "r") as file:
            # Perform operations on the file
            content = file.read()
    # If the file is found and successfully opened, continue here
    except FileNotFoundError:
        # Handle the FileNotFoundError
        fileChecker = 0
        print("Data file not found. Please wait for the completion of the whole program.")
    except Exception as e:
        # Handle other exceptions if necessary
        fileChecker = 0
        print(f"An error occurred: {e}")

    # Get data from "data.txt"
    if fileChecker == 1:
        magnitude_angle_density, min_max_density = dataGetter(PATH)
    
    
    # Initialize the info tracker of each people
    track_points = {}
    track_origin_points = {}
    track_colors = {}
    track_rho = {}
    track_phi = {}
    track_xy_prime = {}

    ## set up input type and classes
    cap = cv2.VideoCapture(input_vid) # input could be video or camera 2
    file = open("CrowdTracking/Kaizhi's/classes.txt","r")
    classes = file.read().split('\n')
    print(classes)
    
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
        
        img_width, img_height = img.shape[1], img.shape[0]
        
        
        # homography process
        # calculate the bottom left and right coordinates of homography
        pt1 = top_left
        pt2 = (17,25)
        m1, n1 = getMN(pt1, pt2)
        bot_left = getOutRangeXY(m1, n1, img_height)
        
        # calculate the bottom left and right coordinates of homography
        pt3 = top_right
        pt4 = (1837,43)
        m2, n2 = getMN(pt3, pt4)
        bot_right = getOutRangeXY(m2, n2, img_height)
        
        # update source coordinates and get image transformed
        srcCoordinates = np.array([top_left, top_right, bot_right, bot_left], np.float32) 
        hMatrix = getTransformed(img, srcCoordinates, False)
        img_trans = getTransformed(img, srcCoordinates, True)
        
        # prepare the birdseye view background
        img_trans_width, img_trans_height = img_trans.shape[1], img_trans.shape[0]
        img_background = np.zeros((img_trans_height, img_trans_width,3),np.uint8)
        img_background[:] = [255, 153, 51]
        
        # draw the forbidden area in the img and birdseye view
        cv2.polylines(img, [forbidden_area], True, (0, 0, 255), 3)
        forbidden_area_trans = forbiddenAreaTrans(forbidden_area, hMatrix)
        cv2.polylines(img_background, [forbidden_area_trans], True, (0, 0, 255), 5)
        
        # build mask image for invade detection
        mask_img = cv2.cvtColor(np.ones_like(img_background),cv2.COLOR_BGR2GRAY)
        mask_img[forbidden_area_trans[1][1]:forbidden_area_trans[3][1], forbidden_area_trans[1][0]:forbidden_area_trans[3][0]] = 0
        
        # generate blob and load it to get detections
        blob = cv2.dnn.blobFromImage(img, 1/255, (640,640), [0,0,0], True, False)
        net.setInput(blob)
        detections = net.forward()[0]
        
        # class_ids, confidences, boxes --need to find
        classes_ids = []
        confidences = []
        boxes = []
        
        rows = detections.shape[0] # each detection
        
        # data used to restore bounding boxes to original size for img
        x_scale = img_width/640
        y_scale = img_height/640
        
        
        # frame_idx=frame_idx+1
        
        
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
                    x1 = int((cx - w/2)*x_scale)
                    y1 = int((cy - h/2)*y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)
                    box = np.array([x1,y1,width,height])
                    boxes.append(box)
        
        # remove the duplicate and overlapping bounding boxes
        NMBoxes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.7)
        
        
        # DeepSORT -> Extracting Bounding boxes and its confidence scores.
        bboxes = []
        scores = []
        
        # show boxes and info
        for i in NMBoxes:
                x1,y1,w,h = boxes[i]
                
                if len(classes) == 1:
                    label = classes[0]
                else:
                    label = classes[classes_ids[i]]
                # conf = confidences[i]
                # text = label + "  {:.2f}".format(conf)
                # color = [int(c) for c in COLORS[classes_ids[i]]]
                if label == 'person':
                    # cv2.rectangle(img,(x1,y1),(x1+w,y1+h),color,2)
                    # cv2.putText(img, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,color,2)
                    
                    # DeepSORT -> Extracting Bounding boxes and its confidence scores.
                    conf = confidences[i]
                    box = [x1, y1, w, h]
                    bboxes.append(box)
                    scores.append(conf)
        
        # DeepSORT -> Getting appearence features of the object.
        features = encoder(img, bboxes)
        # DeepSORT -> Storing all the required info in a list.
        dets = [Detection(bbox, score, feature) for bbox, score, feature in zip(bboxes, scores, features)]
        
        # DeepSORT -> Predicting Tracks. 
        tracker.predict()
        tracker.update(dets)
        
        ## random colors for bounding boxes and tags
        numPeople = len(tracker.tracks)
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(numPeople, 3), dtype='uint8')
        colorCount = 0
        # DeepSORT -> Plotting the tracks.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
    
            # DeepSORT -> Changing track bbox to top left, bottom right coordinates
            bbox = list(track.to_tlbr())
            
            # Extract person's info
            person_id = track.track_id
            color = [int(c) for c in COLORS[colorCount]]
            
            txt = 'id:' + str(person_id)
            if person_id in track_colors:
                color = track_colors[person_id]
            
            
            
            # Homography Transform -> Transform poeple's coordinates and write ID on it
            people_position = getCoordinate(bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1]))
            people_x = int(people_position[0])
            people_y = int(people_position[1])
            trasformed_position = np.dot(hMatrix, people_position)
            newX, newY, temp = trasformed_position/trasformed_position[2]
            newX = int(newX)
            newY = int(newY)
            # detect if person is in the warning area
            if maskImgDetect(mask_img, (newX, newY)):
                color = (0,0,255)
            
            # draw points on birdseye view canvas
            cv2.circle(img_background, (newX, newY),15,color,30)
            cv2.putText(img_background, txt, (newX, newY), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 5)
            

            # draw the tracklet of each person
            if person_id in track_points.keys():
                track_origin_points[person_id].append((people_x, people_y))
                track_points[person_id].append((newX, newY))
                coordinates = track_points[person_id]
                coordinates_origin = track_origin_points[person_id]
                prev_x, prev_y = coordinates[len(coordinates)-2]
                prev_origin_x, prev_origin_y = coordinates_origin[len(coordinates_origin)-2]
                # Normal Distribution -> step1
                x_prime = newX - prev_x
                y_prime = newY - prev_y
                track_xy_prime[person_id].append((x_prime, y_prime))
                # Normal Distribution -> step2
                r, p = coordinateToPolar(x_prime, y_prime)
                track_rho[person_id].append(r)
                track_phi[person_id].append(p)
                # Estimate trajectory
                if fileChecker == 1:
                    kde_density = findDensityByMA(r, p, magnitude_angle_density)
                    if kde_density != -1:
                        if kde_density > anomly_threshold:
                            # kde_color = value_to_color(kde_density, min_max_density[0], min_max_density[1], colorBlue2Red())
                            kde_color = [132, 128, 253]
                            # draw points on original image
                            cv2.circle(img, (people_x, people_y),5,kde_color,10)
                        else:
                            # kde_color = value_to_color(kde_density, min_max_density[0], min_max_density[1], colorBlue2Red())
                            kde_color = [253, 2, 86]
                            # draw points on original image
                            cv2.circle(img, (people_x, people_y),5,kde_color,10)

                for i in range(1, len(coordinates)):
                    prev_x, prev_y = coordinates[i-1]
                    current_x, current_y = coordinates[i]
                    prev_origin_x, prev_origin_y = coordinates_origin[i-1]
                    current_origin_x, current_origin_y = coordinates_origin[i]
                    cv2.line(img_background, (int(prev_x), int(prev_y)), (int(current_x), int(current_y)), color, 4)
                    cv2.line(img, (int(prev_origin_x), int(prev_origin_y)), (int(current_origin_x), int(current_origin_y)), kde_color, 4)
                    
            else:
                track_points[person_id] = ([(newX, newY)])
                track_colors[person_id] = [int(c) for c in COLORS[colorCount]]
                track_rho[person_id] = []
                track_phi[person_id] = []
                track_xy_prime[person_id] = []
                track_origin_points[person_id] = [(people_x, people_y)]
                
            colorCount += 1
            
            

            # DeepSORT -> Writing Track bounding box and ID on the frame using OpenCV.
            (label_width,label_height), baseline = cv2.getTextSize(txt , cv2.FONT_HERSHEY_SIMPLEX,1,1)
            top_left_box = tuple(map(int,[int(bbox[0]),int(bbox[1])-(label_height+baseline)]))
            top_right_box = tuple(map(int,[int(bbox[0])+label_width,int(bbox[1])]))
            org = tuple(map(int,[int(bbox[0]),int(bbox[1])-baseline]))
    
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
            cv2.rectangle(img, top_left_box, top_right_box, color, -1)
            cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            # DeepSORT -> Saving Track predictions into a text file.
            # save_format = '{frame},{id},{x1},{y1},{w},{h},{x},{y},{z}\n'
            #
            # with open(str(Path(save_path).with_suffix('.txt')), 'a') as f:
            #     line = save_format.format(frame=frame_idx, id=track.track_id, x1=int(bbox[0]), y1=int(bbox[1]), w=int(bbox[2]- bbox[0]), h=int(bbox[3]-bbox[1]), x = -1, y = -1, z = -1)
            #     f.write(line)
        
        
        
        # show img
        threeImgShow(img_trans, img_background, img, videoSaver)
        
        k = cv2.waitKey(300) # 5 seconds = 5 * 1000
        if k == ord('q'):
            break
    
    # Normal Distribution -> step3
    total_rho = combineLists(track_rho.values())
    total_phi = combineLists(track_phi.values())
    mean_magnitude = sum(total_rho) / len(total_rho)
    mean_angle = sum(total_phi) / len(total_phi)
    std_magnitude = np.std(total_rho)
    std_angle = np.std(total_phi)

    if fileChecker != 1:
        # Calculate the KDE density values for whole data
        kde_density_values = getKdeValue(total_rho, total_phi)

        # Store Magnitude, Angle, and KDE density values in order seperated by "\t" in "data.txt"
        dataSaver(total_rho, total_phi, kde_density_values, PATH)
    
    print('Mean(magnitude) = ' + str(mean_magnitude) + '    Mean(angle) = ' + str(mean_angle) + '\n')
    print('Std(magnitude) = ' + str(std_magnitude) + '    Std(angle) = ' + str(std_angle) + '\n')
    
    plotGraphs(total_rho, total_phi, mean_magnitude, std_magnitude, mean_angle, std_angle)

    vw.release()
    cap.release()
    cv2.destroyAllWindows()

