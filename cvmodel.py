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

    
    new_width = int(width * scale/100.0)
    new_height = int(height * scale/100.0)
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
    imgs = []
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w<20 or h<20:
            continue
        #cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        ROI = original[y:y+h, x:x+w]
        roi_tensor = preprocess(ROI)

        #get prediction
        with torch.no_grad():
            output = model(roi_tensor)
            _, predicted = torch.max(output, 1)
            prediction_label = label_encoder.inverse_transform([predicted.item()])[0]

        #print(predicted)
        #print(prediction_label)

        #cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        ROI_number += 1
        imgs.append(ROI)

        mid_box_coords = [(x + (x+w))/2, (y + (y+h))/2]
        top_left = [x, y]

        #cv2.rectangle(image, (502, 301), (588, 499), (255, 12, 36), 2)
        #cv2.rectangle(image, (1201, 452), (1249, 500), (255, 12, 36), 2)
        
        imgs_info.append({'category': prediction_label, 
                         'bbox': top_left
                        })


    #image = cv2.circle(image, (100,459), radius=6, color=(255, 12, 36), thickness=10)
    #image = cv2.circle(image, (467,461), radius=6, color=(255, 12, 36), thickness=10)
    #image = cv2.circle(image, (318, 461), radius=6, color=(255, 12, 36), thickness=10)
    #image = cv2.circle(image, (368,461), radius=6, color=(255, 12, 36), thickness=10)
    #image = cv2.circle(image, (301,251), radius=6, color=(255, 12, 36), thickness=10)

    #cv2.imshow('image', image)
    #cv2.waitKey()
    #cv2.imshow('gray', gray)
    #cv2.waitKey()
    #cv2.imshow('threshold', thresh)
    #cv2.waitKey()

    #print(imgs_info)
    
    return image, imgs_info, ROI_number, imgs

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

def split_roi(roi, orig_height, orig_width, og_coords):
    """
    Split ROI into sub-regions if multiple objects are detected
    """
    print("split roi")
    print(f"orig height: {orig_height}, orig width: {orig_width}")
    edge_mask = detect_object_boundaries(roi)
    
    cnts, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected=False
    
    if len(cnts) <= 1:
        return [roi], [(0,0)], [og_coords], detected
    
    detected = True
    
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    split_rois = []
    modified_roi = roi.copy()

    sub_roi_top_left_coords = []

    sub_roi_coords = []

    global_coords = []

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
        global_coords.append(og_coords)
        
        modified_roi[y_start:y_end, x_start:x_end] =255

    if np.all(np.isin(modified_roi, [255, 220])):
        return split_rois, sub_roi_top_left_coords, global_coords, detected

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
        global_coords.append(og_coords)
    
    return split_rois, sub_roi_top_left_coords, global_coords, detected

    #for i, roi in enumerate(split_rois):
        #if (i==0):
         #   roi_filename = 'ROI_{}.png'.format(j[0])
        #else:
         #   roi_filename = "ROI_{}.png".format(i-1+rois)
            
        #print('writing file to ', roi_filename)
        #print(roi)
        #cv2.imwrite(roi_filename, roi)

def classify_rois(rois, top_left_coords, og_coords, model, label_encoder, image):
    """
    Classify each split ROI
    """
    print("classify_rois")

    imgs_info = []
    target_size = (32,32)

    #print(len(rois))
    #print(len(top_left_coords))
    
    for i, roi in enumerate(rois):
        # Preprocess ROI for classification
        if roi.shape[0] != target_size[0] or roi.shape[1] != target_size[1]:
            roi_resized = cv2.resize(roi, target_size)
        else:
            roi_resized = roi

        roi_height, roi_width = roi.shape[:2]
        roi_tensor = preprocess(roi_resized)

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
                    for z in range(int(dup)):
                        sub_roi = roi[banana_height*z:banana_height*(z+1),:]
                        height = banana_height*z
                        width = 0
                        h, w, _ = sub_roi.shape
                        mid_box_coords = [height, width]
                        x = int(((og_coords[i][0]+width)*2+w)/2)
                        y = int(((og_coords[i][1]+height)*2+h)/2)
                        new_box_coords = [x, y]

                        cv2.rectangle(image, (x, y), (x + w, y + h), (255,12,36), 2)

                        # Store object information
                        imgs_info.append({
                            'category': prediction_label, 
                            'bbox': new_box_coords,
                            'roi': sub_roi
                        })
                if roi_width >= banana_width*2:
                    print("hit2")
                    dup = roi_width / banana_width
                    print(dup)
                    for j in range(int(dup)):
                        sub_roi = roi[:,banana_width*j: banana_width*(j+1)]
                        height = 0
                        width = banana_width*j
                        h, w, _ = sub_roi.shape
                        mid_box_coords = [height, width]
                        x = int(((og_coords[i][0]+width)*2+w)/2)
                        y = int(((og_coords[i][1]+height)*2+h)/2)
                        new_box_coords = [x, y]

                        cv2.rectangle(image, (x, y), (x + w, y + h), (255,12,36), 2)

                        # Store object information
                        imgs_info.append({
                            'category': prediction_label, 
                            'bbox': new_box_coords,
                            'roi': sub_roi
                        })

        # Compute midpoint
        #print('i: ', i)
        width, height = top_left_coords[i]

        if prediction_label != 'monkey' and prediction_label != 'banana':
            width = width-16
            height = height-10
        h, w, _ = roi.shape
        x = int(((width+og_coords[i][0])*2+w)/2)
        y = int(((height+og_coords[i][1])*2+h)/2)
        mid_box_coords = [x, y]
        #print(x)
        #print(y)
        #print(w)
        #print(h)
        
        cv2.rectangle(image, (width+og_coords[i][0], height+og_coords[i][1]), 
                      (og_coords[i][0]+width + w, og_coords[i][1]+height + h), (255,12,36), 2)

        # Store object information
        imgs_info.append({
            'category': prediction_label, 
            'bbox': mid_box_coords,
            'roi': roi
        })
    
    return imgs_info, image

def process_roi(roi, model, label_encoder, info, coords, image):
    """
    Main processing function for a single ROI
    """
    print("process_roi")
    # Read the ROI image
    #roi = cv2.imread(img_path)
    height, width, _ = roi.shape
    og_coords = coords
    
    # Split ROI if multiple objects detected
    split_rois, top_left_coords, og_coords, detected = split_roi(roi, height, width, og_coords)

    
    # Classify split ROIs
    imgs_info, image = classify_rois(split_rois, top_left_coords,og_coords, model, label_encoder, image)

    
    
    #Optional: visualize split ROIs
    #for i, info in enumerate(imgs_info):
        #print(info)
        #cv2.imshow(f'ROI {i+1} - {info["category"]}', info['roi'])
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return imgs_info, image

if __name__ == "__main__":
    model = load_model()
    img_path = 'BTAI_genImages_extra\canvas_0_banana2_monkey1_box4.png'
    label_encoder = joblib.load('label_encoder.pkl')
    result_img, info, rois, imgs = process_img(img_path, model)

    new_info=[]
  
    
    for i in enumerate(info):
        roi = imgs[i[0]]
        #print(img_path)
        
        #print(info[i[0]]['bbox'])
        results, image = process_roi(roi, model, label_encoder, info, i[1]['bbox'], result_img)
        for k, detection in enumerate(results):
            new_info.append(results[k])
        
    #print(type(info))
    #print(type(new_info))

    final_result = []
    cv2.imshow('image', image)
    cv2.waitKey()

    

    for i, detection in enumerate(new_info):
        print(f"Object {i+1}:")
        print(f"Category: {detection['category']}")
        print(f"Bounding Box: {detection['bbox']}")

        final_result.append({
            'category': detection['category'],
            'bbox': detection['bbox']
        })

