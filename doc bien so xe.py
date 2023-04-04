## LOAD THU VIEN VA MODUL CAN THIET
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

######## Upload KNN model ######################
npaClassifications = np.loadtxt("classificationS.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
npaClassifications = npaClassifications.reshape(
    (npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train
kNearest = cv2.ml.KNearest_create()  # instantiate KNN object
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
#----------------------DOC HINH ANH - TACH HINH ANH NHAN DIEN--------------------
img = cv2.imread('11.jpg')
cv2.imshow('HINH ANH GOC', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
contours,h = cv2.findContours(thresh,1,2)
largest_rectangle = [0,0]
for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    if len(approx)==4: 
        area = cv2.contourArea(cnt)
        if area > largest_rectangle[0]:
            largest_rectangle = [cv2.contourArea(cnt), cnt, approx]
x,y,w,h = cv2.boundingRect(largest_rectangle[1])

image=img[y:y+h,x:x+w]
cv2.drawContours(img,[largest_rectangle[1]],0,(0,255,0),3)

cropped = img[y-3:y+h+1, x+4:x+w-1]
cv2.imshow('DANH DAU DOI TUONG', img)

cv2.drawContours(img,[largest_rectangle[1]],0,(255,255,255),18)


#--------------------- DOC HINH ANH CHUYEN THANH FILE TEXT-----------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imshow('CROP', thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
invert = 255 - opening

cv2.waitKey()
