import cv2 as cv
import numpy as np

def resize(img, percent):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height) 
    return cv.resize(img, dim, interpolation = cv.INTER_AREA) 

def func(file1, file2):
    img1 = cv.imread(file1, cv.IMREAD_GRAYSCALE)
    img2 = resize(cv.imread(file2, cv.IMREAD_GRAYSCALE), 15)

    # ORB Detector
    #orb = cv2.ORB_create()
    akaze = cv.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    # Brute Force Matching
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    return cv.drawMatches(img1, kp1, img2, kp2, matches[:70], None, flags=2)
    
    


file1 = "origin.jpg"
for i in range(54, 60):
    file2 = "photo/book"+str(i)+".jpg"
    cv.imshow("Matching result"+str(i), func(file1, file2))

cv.waitKey(0)
cv.destroyAllWindows()


