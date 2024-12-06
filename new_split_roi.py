import cv2
import torch
import torchvision.transforms as transforms
import joblib
from PIL import Image
import torch.nn as nn
import numpy as np
import os

def preprocess(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(pil_img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def load_model():
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(128 * 4* 4, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 7)
    )
    model.load_state_dict(torch.load('cnn.pth'))
    model.eval()
    return model

def detect_object_boundaries(roi):
    """
    Detect object boundaries using Sobel filters and morphological operations
    """
    print("object boundaries")
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    _, edge_mask = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
    
    return edge_mask

def split_roi(roi, orig_height, orig_width):
    """
    Split ROI into sub-regions if multiple objects are detected
    """
    print("split roi")
    print(f"orig height: {orig_height}, orig width: {orig_width}")
    edge_mask = detect_object_boundaries(roi)
    
    cnts, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) <= 1:
        return [roi], [(0,0)]
    
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    split_rois = []
    modified_roi = roi.copy()

    sub_roi_top_left_coords = []

    sub_roi_coords = []

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        
        if w < 10 or h < 10:
            continue
        
        sub_roi = roi[y:y+h, x:x+w]    
        if sub_roi.shape == roi.shape:
            continue
        
        x_start = max(int(x - w-4), 0)
        y_start = max(int(y - h-4), 0)
        x_end = min(int(x + w + w+4), roi.shape[1])
        y_end = min(int(y + h + h+4), roi.shape[0])
        if y_end +10 > orig_height:
            y_end = orig_height
        if x_end +10 > orig_width:
            x_end = orig_width
        if x_start - 10 < 0:
            x_start = 0
        if y_start - 10 < 0:
            y_start = 0

        sub_roi = roi[y_start:y_end, x_start:x_end]    
        split_rois.append(sub_roi)

        #entered in order of 'mid_x', 'mid_y', x_start, x_end, y_start, y_end
        sub_roi_coords.append([x + w / 2, y + h / 2,x_start,x_end,y_start,y_end])
        sub_roi_top_left_coords.append((x,y))
        
        modified_roi[y_start:y_end, x_start:x_end] =255

    if np.all(np.isin(modified_roi, [255, 220])):
        return split_rois, sub_roi_top_left_coords

    to_remove = set()

    #modified roi coordinates (assume top left corner of roi input is (0,0))
    m_roi_x = 0
    m_roi_y = 0
    for i in range(len(sub_roi_coords)-1):
        for j in range(i+1,len(sub_roi_coords)):
            mid_x1, mid_y1, x_start1, x_end1, y_start1, y_end1 = sub_roi_coords[i]
            mid_x2, mid_y2,x_start2,x_end2,y_start2,y_end2 = sub_roi_coords[j]

            # Horizontal alignment (same y-coordinate)
            if abs(mid_y1 - mid_y2) <= 3:
                print("horizontal hit")
                if x_end1 == orig_width or x_end2 == orig_width: 
                    if x_start1 == 0 or x_start2 == 0:
                        if y_start1-10<0:
                            modified_roi = modified_roi[y_end1:, :] 
                            m_roi_y = y_end1
                            to_remove.add(i)
                            to_remove.add(j)
                            break
                        elif y_end1+10>orig_height:
                            modified_roi = modified_roi[:y_start1, :] 
                            to_remove.add(i)
                            to_remove.add(j)
                            break
            # Vertical alignment (same x-coordinate)
            elif abs(mid_x1 - mid_x2) <=3:
                if y_end1 == orig_height or y_end2 == orig_height: 
                    if y_start1 == 0 or y_start2 == 0:
                        if x_start1-10<0:
                            modified_roi = modified_roi[:, x_end1:] 
                            m_roi_x = x_end1
                            to_remove.add(i)
                            to_remove.add(j)
                            break
                        elif x_end +10> orig_width:
                            modified_roi = modified_roi[:, :x_start1] 
                            to_remove.add(i)
                            to_remove.add(j)
                            break
    
    sub_roi_coords = [coords for i, coords in enumerate(sub_roi_coords) if i not in to_remove]    
    if sub_roi_coords:
        for i,coords in enumerate(sub_roi_coords):
            mid_x,mid_y,x_start,x_end,y_start,y_end = coords
            if y_start > 10:
                modified_roi = modified_roi[:y_start, :]
            elif x_end < (orig_width-10):
                modified_roi = modified_roi[:,x_end:]
                m_roi_x = x_end
            elif x_start > 10:
                modified_roi = modified_roi[:,:x_start]
            elif y_end < (orig_height-10):
                modified_roi = modified_roi[y_end:,:]
                m_roi_y = y_end
    m_height, m_width = modified_roi.shape[:2]
    area = m_height * m_width
    if area > 20:
        split_rois.append(modified_roi)
        sub_roi_top_left_coords.append((m_roi_x, m_roi_y))
    
    return split_rois, sub_roi_top_left_coords

def classify_rois(rois, top_left_coords, model, label_encoder):
    """
    Classify each split ROI
    """
    print("classify_rois")

    imgs_info = []
    target_size = (32,32)
    
    for i,roi in enumerate(rois):
        # Preprocess ROI for classification
        if roi.shape[0] != target_size[0] or roi.shape[1] != target_size[1]:
            roi_resized = cv2.resize(roi, target_size)
        else:
            roi_resized = roi
        roi_height, roi_width = roi.shape[:2]
        roi_tensor = preprocess(roi_resized)

        # Classification
        with torch.no_grad():
            output = model(roi_tensor)
            _, predicted = torch.max(output, 1)
            prediction_label = label_encoder.inverse_transform([predicted.item()])[0]

            #handle banana case
            #singular banana is 35 x 50 (height x width)
            if prediction_label == 'banana':
                banana_height = 35
                banana_width = 50
                print(f"{roi_height, roi_width}")
                if roi_height >= banana_height*2:
                    print("hit")
                    dup = roi_height / banana_height
                    for i in range(int(dup)):
                        sub_roi = roi[banana_height*i:banana_height*(i+1),:]
                        height = banana_height*i
                        width = 0
                        mid_box_coords = [height, width]

                        # Store object information
                        imgs_info.append({
                            'category': prediction_label, 
                            'bbox': mid_box_coords,
                            'roi': sub_roi
                        })
                if roi_width >= banana_width*2:
                    print("hit2")
                    dup = roi_width / banana_width
                    print(dup)
                    for i in range(int(dup)):
                        sub_roi = roi[:,banana_width*i: banana_width*(i+1)]
                        height = 0
                        width = banana_width*i
                        mid_box_coords = [height, width]
                        

                        # Store object information
                        imgs_info.append({
                            'category': prediction_label, 
                            'bbox': mid_box_coords,
                            'roi': sub_roi
                        })
                    continue

        # Compute midpoint
        width, height = top_left_coords[i]
        mid_box_coords = [height, width]

        # Store object information
        imgs_info.append({
            'category': prediction_label, 
            'bbox': mid_box_coords,
            'roi': roi
        })
    
    return imgs_info

def process_roi(img_path, model, label_encoder):
    """
    Main processing function for a single ROI
    """
    print("process_roi")
    # Read the ROI image
    roi = cv2.imread(img_path)
    height, width,_ = roi.shape
    
    # Split ROI if multiple objects detected
    split_rois, top_left_coords = split_roi(roi, height, width)
    
    # Classify split ROIs
    imgs_info = classify_rois(split_rois, top_left_coords, model, label_encoder)
    
    # Optional: visualize split ROIs
    for i, info in enumerate(imgs_info):
        cv2.imshow(f'ROI {i+1} - {info["category"]}', info['roi'])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return imgs_info

# Load model and label encoder
model = load_model()
label_encoder = joblib.load('label_encoder.pkl')

# Replace this with the path to your ROI image
folder_path = 'new_pair_imgs'
image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png')]

roi_folder = []
for i,img_path in enumerate(image_files):
    print(f"picture {i}")
    results = process_roi(img_path, model, label_encoder)

    # Print detection results
    for i, detection in enumerate(results):
        print(f"Object {i+1}:")
        print(f"Category: {detection['category']}")
        print(f"Bounding Box: {detection['bbox']}")
