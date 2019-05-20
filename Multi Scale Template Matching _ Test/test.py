# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:37:21 2019

@author: ADARSH
"""

# Python program to illustrate 
# template matching 
import cv2 
import numpy as np 
import csv
import matplotlib.pyplot as plt

'''
# Read the main image 
img_rgb = cv2.imread('main1.jpg') 

# Convert it to grayscale 

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) 


# Read the template 
template = cv2.imread('template1.jpg',0) 
#template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)


# Store width and heigth of template in w and h 
w, h = template.shape[::-1] 

# Perform match operations. 
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED) 
print(res)
# Specify a threshold 
threshold = 0.7

# Store the coordinates of matched area in a numpy array 
loc = np.where( res >= threshold)
print(loc)
 

# Draw a rectangle around the matched region. 
for pt in zip(*loc[::-1]): 
	cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2) 

# Show the final image with the matched area. 

cv2.imshow('Detected',img_rgb) 
cv2.waitKey()
'''

def template_scaling(image_path,des):
 
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
 
    print('Original Dimensions : ',img.shape)


    scale_percent = np.array([15,25,40,50,70,100,120], dtype =np.float64) # percent of original size

    for x in range(7):
        width = int(img.shape[1] * scale_percent[x] / 100)
        height = int(img.shape[0] * scale_percent[x] / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        print('Resized Dimensions : ',resized.shape)
        name = des +str(x) + '.jpg'
        cv2.imwrite(name, resized)
        
template_scaling('template1.jpg','temp')
template_scaling('main1.jpg','m')


def multi_scale_template_matching(template, main):
    
    for m in main:
        #cv2.imshow('image',m)
        #cv2.waitKey()
        
        img_rgb = m 
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        imw, imh = img_gray.shape[::-1]
        tot = imw*imh
        for t in template:
            #cv2.imshow('image',t)
            #cv2.waitKey()
            tmp = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
            w, h = tmp.shape[::-1] 
            tot2 = w*h
            if tot2 < tot:
                
                match = cv2.matchTemplate(img_gray,tmp,cv2.TM_CCOEFF_NORMED)
                threshold = 0.75
                loc = np.where( match >= threshold)
                for pt in zip(*loc[::-1]): 
                    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
                
                #print(match)
                    cv2.imshow('Detected',img_rgb) 
                    cv2.waitKey()
                    
             

templates = []
main_images = []
for y in range(7):
    name_templates = 'temp' +str(y) + '.jpg'
    name_main = 'm' +str(y) + '.jpg'
    templates.append(plt.imread(name_templates))    
    main_images.append(plt.imread(name_main))
#templates = np.array(templates, dtype = 'float32')
#main_images = np.array(main_images, dtype = 'float32')
    
        
multi_scale_template_matching(templates, main_images)





