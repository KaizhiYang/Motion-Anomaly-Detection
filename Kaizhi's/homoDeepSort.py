import cv2
import numpy as np 
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
vidFile = "CrowdTracking/Kaizhi's/result.mp4"
vw = cv2.VideoWriter(vidFile, fourcc, 24, (2302,1302),1) # resolution has to align with the image you want to save as a video
videoSaver = 1

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

def threeImgShow(img1, img2, img3):
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
    if videoSaver == 1:
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
    
    
if __name__ == '__main__':
    # DeepSORT -> Intializing tracker.
    max_cosine_distance = 0.4
    nn_budget = None
    model_filename = "CrowdTracking/Kaizhi's/mars-small128.pb"
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    
    # frame_idx=0
    
    # Initialize the position and color of each people
    track_points = {}
    track_colors = {}
    
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
            colorCount += 1
            txt = 'id:' + str(person_id)
            if person_id in track_colors:
                color = track_colors[person_id]
            
            
            
            # Homography Transform -> Transform poeple's coordinates and write ID on it
            people_position = getCoordinate(bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1]))
            trasformed_position = np.dot(hMatrix, people_position)
            newX, newY, temp = trasformed_position/trasformed_position[2]
            
            # detect if person is in the warning area
            if maskImgDetect(mask_img, (newX, newY)):
                color = (0,0,255)
            
            cv2.circle(img_background, (int(newX), int(newY)),15,color,30)
            cv2.putText(img_background, txt, (int(newX), int(newY)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 5)
            
            # draw the tracklet of each person
            if person_id in track_points:
                track_points[person_id].append((newX, newY))
                coordinates = track_points[person_id]
                prev_x, prev_y = coordinates[0]
                for i in range(1, len(coordinates)):
                    current_x, current_y = coordinates[i]
                    cv2.line(img_background, (int(prev_x), int(prev_y)), (int(current_x), int(current_y)), color, 4)
                    prev_x, prev_y = coordinates[i]
            else:
                track_points[person_id] = ([(newX, newY)])
                track_colors[person_id] = [int(c) for c in COLORS[colorCount]]
            
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
        threeImgShow(img_trans, img_background, img)
        
        k = cv2.waitKey(400) # 5 seconds = 5 * 1000
        if k == ord('q'):
            break
        
    vw.release()
    cap.release()
    cv2.destroyAllWindows()

