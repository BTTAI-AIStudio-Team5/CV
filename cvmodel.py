import cv2
import numpy as np

# Load image, grayscale, Otsu's threshold 
image = cv2.imread('BTAI_genImages_150\canvas_3_banana1_monkey1_box3.png')

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
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if w<20 or h<20:
        continue
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    ROI = original[y:y+h, x:x+w]
    cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
    ROI_number += 1


cv2.imshow('image', image)
cv2.waitKey()
cv2.imshow('gray', gray)
cv2.waitKey()
cv2.imshow('threshold', thresh)
cv2.waitKey()
