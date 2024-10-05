import cv2
from matplotlib import pyplot as plt

img = cv2.imread("BTAI_genImages_150\canvas_0_banana1_monkey1_box0.png")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#plot original img
plt.subplot(1,1,1)
plt.imshow(img_rgb)
plt.show()

# using xml cascade classifier
'''
stop_data = cv2.CascadeClassifier()
found = stop_data.detectMultiScale(img_gray, minSize = (20,20))
amount_found = len(found)

if amount_found != 0:
    for(x,y,width,height) in found:
        cv2.rectangle(img_rgb, (x,y), (x+height, y+width), (0,255,0), 5)
'''
#find contours
ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
cv2.imshow("")

