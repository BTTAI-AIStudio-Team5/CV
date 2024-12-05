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



#load model
def load_model():
    global label_encoder
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
    label_encoder = joblib.load('label_encoder.pkl')
    return model

def process_img(image_path, model):
    global label_encoder
    # Load image, grayscale, Otsu's threshold 
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    scale = 50

    
    new_width = int(width * 50.0/100.0)
    new_height = int(height * 50.0/100.0)
    image = cv2.resize(image, (new_width, new_height))

    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #gray_filtered = cv2.inRange(gray, 111, 175)
    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 1)
    thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)[1]
    #+ cv2.THRESH_OTSU

    #255 = white
    #less =darker, more=lighter

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255,255,255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
    result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

    #cv2.imshow('thresh', thresh)
    #cv2.imshow('detected_lines', detected_lines)
    #cv2.imshow('image', image)
    #cv2.imshow('result', result)
    #cv2.waitKey()

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)[1]
    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 1)

    # Find contours, obtain bounding box, extract and save ROI
    ROI_number = 0
    imgs_info = []
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w<20 or h<20:
            continue
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        ROI = original[y:y+h, x:x+w]
        roi_tensor = preprocess(ROI)

        #get prediction
        with torch.no_grad():
            output = model(roi_tensor)
            _, predicted = torch.max(output, 1)
            prediction_label = label_encoder.inverse_transform([predicted.item()])[0]

        #print(predicted)
        #print(prediction_label)

        cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        ROI_number += 1

        mid_box_coords = [(x + (x+w))/2, (y + (y+h))/2]

        #cv2.rectangle(image, (502, 301), (588, 499), (255, 12, 36), 2)
        #cv2.rectangle(image, (1201, 452), (1249, 500), (255, 12, 36), 2)
        
        imgs_info.append({'category': prediction_label, 
                         'bbox': mid_box_coords
                        })
        
    #cv2.imshow('image', image)
    #cv2.waitKey()
    #cv2.imshow('gray', gray)
    #cv2.waitKey()
    #cv2.imshow('threshold', thresh)
    #cv2.waitKey()

    #print(imgs_info)
    
    return image, imgs_info, ROI_number

def detect_object_boundaries(roi):
    """
    Detect object boundaries using Sobel filters and morphological operations
    """
    print("object boundaries")
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Sobel X and Y
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine and normalize Sobel edges
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Threshold to get strong edges
    _, edge_mask = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up edges
    kernel = np.ones((3,3), np.uint8)
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
    
    return edge_mask

def detect_object_boundaries(roi):
    """
    Detect object boundaries using Sobel filters and morphological operations
    """
    print("object boundaries")
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Sobel X and Y
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine and normalize Sobel edges
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Threshold to get strong edges
    _, edge_mask = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up edges
    kernel = np.ones((3,3), np.uint8)
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
    
    return edge_mask

def split_roi(roi, model, label_encoder, rois, j, info):
    """
    Split ROI into sub-regions if multiple objects are detected
    """
    print("split roi")
    # Detect boundaries
    edge_mask = detect_object_boundaries(roi)
    
    # Find contours in the edge mask
    cnts, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no clear separation or too many contours, return the original ROI
    if len(cnts) <= 1:
        return [roi], rois
    
    # Sort contours by area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # Store split ROIs
    split_rois = []
    overlap_factor = 1

    # Extract sub-regions based on contours
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Skip very small regions
        if w < 10 or h < 10:
            continue
        
        sub_roi = roi[y:y+h, x:x+w]    
        if sub_roi.shape == roi.shape:
            continue        

        x_overlap = int(w * overlap_factor)
        y_overlap = int(h * overlap_factor)

        x_start = max(x - x_overlap, 0)
        y_start = max(y - y_overlap, 0)
        x_end = min(x + w + x_overlap, roi.shape[1])
        y_end = min(y + h + y_overlap, roi.shape[0])

        sub_roi = roi[y_start:y_end, x_start:x_end]    
        split_rois.append(sub_roi)

    for i, roi in enumerate(split_rois):
        if (i==0):
            print(roi)
            roi_filename = 'ROI_{}.png'.format(j[0])
            info[j[0]]= i
            print(info[j[0]])
        else:
            roi_filename = "ROI_{}.png".format(i-1+rois)
            info.append(i)
            
        print('writing file to ', roi_filename)
        cv2.imwrite(roi_filename, roi)

    return split_rois, rois+1

def classify_rois(rois, model, label_encoder):
    """
    Classify each split ROI
    """
    print("classify_rois")

    imgs_info = []
    target_size = (32,32)
    
    for roi in rois:
        # Preprocess ROI for classification
        if roi.shape[0] != target_size[0] or roi.shape[1] != target_size[1]:
            roi_resized = cv2.resize(roi, target_size)
        else:
            roi_resized = roi

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
                img = cv2.imread(img_path)
                height, width,_ = img.shape
                print(f"{height, width}")
                if height > banana_height:
                    print("hit")
                    dup = height / banana_height
                    for i in range(int(dup)):
                        sub_roi = roi[banana_height*i:banana_height*(i+1),:]
                        h, w = sub_roi.shape[:2]
                        mid_box_coords = [w/2, h/2]

                        # Store object information
                        imgs_info.append({
                            'category': prediction_label, 
                            'bbox': mid_box_coords,
                            'roi': sub_roi
                        })
                if width > banana_width:
                    print("hit2")
                    dup = width / banana_width
                    print(dup)
                    for i in range(int(dup)):
                        sub_roi = roi[:,banana_width*i: banana_width*(i+1)]
                        h, w = sub_roi.shape[:2]
                        mid_box_coords = [w/2, h/2]

                        # Store object information
                        imgs_info.append({
                            'category': prediction_label, 
                            'bbox': mid_box_coords,
                            'roi': sub_roi
                        })

        # Compute midpoint
        h, w = roi.shape[:2]
        mid_box_coords = [w/2, h/2]

        # Store object information
        imgs_info.append({
            'category': prediction_label, 
            'bbox': mid_box_coords,
            'roi': roi
        })
    
    return imgs_info

def process_roi(img_path, model, label_encoder, rois, j, info):
    """
    Main processing function for a single ROI
    """
    print("process_roi")
    # Read the ROI image
    roi = cv2.imread(img_path)
    
    # Split ROI if multiple objects detected
    split_rois, new_rois = split_roi(roi, model, label_encoder, rois, j, info)
    
    # Classify split ROIs
    imgs_info = classify_rois(split_rois, model, label_encoder)
    
    # Optional: visualize split ROIs
    for i, info in enumerate(imgs_info):
        cv2.imshow(f'ROI {i+1} - {info["category"]}', info['roi'])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return imgs_info, new_rois

if __name__ == "__main__":
    model = load_model()
    img_path = 'BTAI_genImages_150\canvas_14_banana1_monkey1_box4.png'
    label_encoder = joblib.load('label_encoder.pkl')
    result_img, info, rois = process_img(img_path, model)
    
    for i in enumerate(info):
        img_path = 'ROI_{}.png'.format(i[0])
        print(img_path)
        results, new_rois = process_roi(img_path, model, label_encoder, rois, i, info)


    for i, detection in enumerate(info):
        print(f"Object {i+1}:")
        print(f"Category: {detection['category']}")
        print(f"Bounding Box: {detection['bbox']}")
