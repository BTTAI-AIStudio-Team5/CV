import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

categories = ["banana", "monkey", "box_A", "box_B", "box_C", "box_D"]

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
    nn.Linear(128, 6)
    )
    model.load_state_dict(torch.load('cnn.pth'))
    model.eval()
    return model

def process_img(image_path, model):
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
            prediction_label = categories[predicted.item()]

        #print(predicted)
        #print(prediction_label)

        cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        ROI_number += 1

        box = [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]

        cv2.rectangle(image, (502, 301), (588, 499), (255, 12, 36), 2)
        #cv2.rectangle(image, (1201, 452), (1249, 500), (255, 12, 36), 2)
        
        imgs_info.append({'category': prediction_label, 
                         'bbox': box
                        })
        
    cv2.imshow('image', image)
    cv2.waitKey()
    #cv2.imshow('gray', gray)
    #cv2.waitKey()
    #cv2.imshow('threshold', thresh)
    #cv2.waitKey()

    #print(imgs_info)
    
    return image, imgs_info

model = load_model()
img_path = 'BTAI_genImages_150\canvas_6_banana2_monkey1_box4.png'
result_img, info = process_img(img_path, model)

for i, detection in enumerate(info):
    print(f"Object {i+1}:")
    print(f"Category: {detection['category']}")
    print(f"Bounding Box: {detection['bbox']}")
