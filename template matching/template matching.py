# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:37:21 2019

@author: ADARSH
"""

# Python program to illustrate 
# template matching 
import cv2 
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import time


def read_templates(root_path):
    
    templates = []
    definition = []
    #root_path = root_path + '/'
    t1 = cv2.imread(root_path+ 'stop_2.jpg',1)
    print(t1.shape)
    #t1 = cv2.imread('stop.jpg',1)
    definition.append('12')
    templates.append(t1)
    t2 = cv2.imread(root_path+'priority_2.jpg',1)
    print(t2.shape)
    templates.append(t2)
    definition.append('11')
    t3 = cv2.imread(root_path+'speed60_2.jpg',1)
    print(t3.shape)
    templates.append(t3)
    definition.append('3')
    t4 = cv2.imread(root_path+'yield_2.jpg',1)
    print(t4.shape)
    templates.append(t4)
    definition.append('13')
    
    return templates, definition

templates, definition = read_templates('C:/traffic sign recognition/Dataset/templates/')

# for reading dataset the classes which we consdering(3,11,12,13), the name of the folders and ground truth csv files has to be renamed from 00000 till 00004
# for first class the folder name is changed from 00003 to 00000 anf csv file name changed from GT-00003.csv to GT-00000.csv
 
    
def readTrafficSigns(rootpath):
   
    images = [] 
    labels = []
   # images_roi = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    
    for c in range(0,4):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
            x1.append(row[3])
            y1.append(row[4])
            x2.append(row[5])
            y2.append(row[6])
       
        gtFile.close()
    print(len(images))
    
    return images, labels, x1, y1, x2, y2

# to read test data, rename the folder to 00000 and csv file to GT-00000.csv
    
def readTrafficSigns_testdata(rootpath):
   
    images = [] 
    test_labels = []
       
    for c in range(0,1):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            test_labels.append(row[7]) # the 8th column is the label
            
        gtFile.close()
    print(len(images))
    # append "0" for all the classes which we are not considering 
    our_class = [3,11,12,13]
    new_labels = []
    for i in range(len(test_labels)):
        
        l = int(test_labels[i])
        if l in our_class:
            k = int(str(test_labels[i]))
        else:
            k = "0"
        new_labels.append(k)
    return images, new_labels

        


                
def multi_scale_template_matching(images,templates):
    
    image = []
    results = []
    
    for m in range(len(images)):
        
        img_1 = images[m]
        
        max_values = []
             
        resize_m = cv2.resize(img_1,(98,98),interpolation = cv2.INTER_AREA)
        
        for t in range (len(templates)):
            #print(t)
            match,max_val = template_matching(resize_m,templates[t])
            max_values.append(max_val)
            #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
         
        resize_m = cv2.resize(img_1,(100,100),interpolation = cv2.INTER_AREA)
        
        for t in range (len(templates)):
            #print(t)
            match,max_val = template_matching(resize_m,templates[t])
            max_values.append(max_val)
            #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
         
        m = max(max_values)
        #print(m)
        ind = max_values.index(m)
        if ind>3 and ind<=7:
            ind = ind-4
        elif ind>7:
            ind = ind-8
        else:
            ind = ind
            
        de = definition[ind]
        
        if m>0.80:
            print(de)
            #print("match_val :", m)
            image.append(m)
            results.append(de)
                        
        else:
            print('0')
            image.append(m)
            results.append("0")
        
    print("matching completed")
    return image, results 



def template_matching(image, template):
    #image = cv2.imread(image,1)
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_canny = Canny(img_gray) # Edge detection using canny egde detector
    ret,thresh_img = cv2.threshold(img_gray,150,255,cv2.THRESH_BINARY)# binary image conversion
    t_img = 255 - thresh_img# negative image conversion
    #print('Original Dimensions -main image : ',img_gray.shape)
    template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    #template = Canny(template)
    ret1,thresh_tmp = cv2.threshold(template,150,255,cv2.THRESH_BINARY)
    t_tmp = 255 - thresh_tmp
    #print('Original Dimensions - template : ',template.shape)
    w,h = template.shape[::-1]
    
    match = cv2.matchTemplate(img_gray, template,cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    #print(max_val)        
    cv2.rectangle(img_gray, max_loc, (max_loc[0]+ w, max_loc[1]+h),(0,0,255),2)
    
    
    
    #cv2.imshow('Detected',img_gray)
    #cv2.waitKey()
    return match,max_val  
            
def accuracy_test(new_labels,results):
    
    accuracy = accuracy_score(new_labels, results)
    accuracy = accuracy*100
    print("Accuracy is : " + str(int(accuracy)) + "%")
   
    #for i in range(len(results)):
    return confusion_matrix(new_labels, results)    
        
                
                    
def Canny (image):
        blur = cv2.GaussianBlur(image,(5,5),0)# for noice reduction
        canny = cv2.Canny(blur, 5 ,50)
        cv2.waitKey()
        
        return canny
                   


'''
start = time.time()
#img, label , roi_x1, roi_y1, roi_x2, roi_y2 = readTrafficSigns('C:/traffic sign recognition/Dataset')
test_images, test_labels = readTrafficSigns_testdata('C:/traffic sign recognition/Dataset/New folder/GTSRB/Final_Test')
start = time.time()
im, results = multi_scale_template_matching(test_images,templates)
accuracy_test(test_labels,results)
end = time.time()
print(end - start)
'''