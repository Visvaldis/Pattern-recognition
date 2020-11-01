import cv2
import numpy as np
import os
import time
import pandas as pd

def resize(img, percent):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height) 
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 

def get_metrics(origin, frameToDetect, threshold = 0.8):

    akaze = cv2.AKAZE_create()

    start = time.time()
    kp_origin, des_origin = akaze.detectAndCompute(origin, None)
    kp_frame, des_frame = akaze.detectAndCompute(frameToDetect, None)

    # Feature matching
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(np.asarray(des_origin,np.float32),np.asarray(des_frame,np.float32), 2)
    good_points = []

    localization_error = 0
    for m, n in matches:
        localization_error += m.distance
        if m.distance < threshold * n.distance:
            good_points.append(m)
    t = time.time() - start
    localization_error /= len(matches)
    relative_num_of_correct_features  = len(good_points) / len(matches)

    return relative_num_of_correct_features, localization_error, t


def get_match_Flann(origin, frameToDetect, threshold = 0.8):
    akaze = cv2.AKAZE_create()
    kp_origin, des_origin = akaze.detectAndCompute(origin, None)
    kp_frame, des_frame = akaze.detectAndCompute(frameToDetect, None)

    # Feature matching
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(np.asarray(des_origin,np.float32),np.asarray(des_frame,np.float32), 2)
    good_points = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_points.append(m)

    return good_points, kp_origin, kp_frame


def homography(good_points, kp_origin, kp_frameToDetect, origin, frameToDetect):
    if len(good_points) > 10:
        query_pts = np.float32([kp_origin[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_frameToDetect[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # Perspective transform
        h, w = origin.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)

        frameToDetect = cv2.cvtColor(frameToDetect, cv2.COLOR_GRAY2BGR)
        homography = cv2.polylines(frameToDetect, [np.int32(dst)], True, (0, 255, 0), 3)
        return homography, True
    else:
        return frameToDetect, False


scores = {
    'm1': [], # відносна к-сть прав суміщ ознак
    'm2': [], # похибка локалізації
    'm3': [], # час
}
names = []
_, __, files = next(os.walk("goose_cup"))
file_count = len(files)

file1 = "goose_cup/origin.jpg"
img1 = resize(cv2.imread(file1, cv2.IMREAD_GRAYSCALE), 50)

#file2 = "2.jpg"
#img2 = resize(cv2.imread(file2, cv2.IMREAD_GRAYSCALE), 50)
#good_points, kp_origin, kp_frameToDetect = get_match_Flann(img1, img2)
#detect_obj, isDetected = homography(good_points, kp_origin, kp_frameToDetect, img1, img2)
#match = cv2.drawMatches(img1, kp_origin, detect_obj, kp_frameToDetect, good_points, None, flags=2)
#cv2.imshow("out/result_" + ("detected" if isDetected else "undefined")  + ".jpg", match)

for i in range(1, file_count):
    try:
        file2 = "goose_cup/goose"+str(i)+".jpg"
        img2 = resize(cv2.imread(file2, cv2.IMREAD_GRAYSCALE), 15)

        good_points, kp_origin, kp_frameToDetect = get_match_Flann(img1, img2)
        detect_obj, isDetected = homography(good_points, kp_origin, kp_frameToDetect, img1, img2)

        #relative_num_of_correct_features, localization_error, times = get_metrics(img1, img2)
        #print(relative_num_of_correct_features, "\t", localization_error)
        #scores['m1'].append(relative_num_of_correct_features)
        #scores['m2'].append(localization_error)
        #scores['m3'].append(times)
        #names.append(file2[:-4])
        match = cv2.drawMatches(img1, kp_origin, detect_obj, kp_frameToDetect, good_points, None, flags=2)
        cv2.imwrite("out_olya/result"+str(i) +"_" + ("detected" if isDetected else "undefined")  + ".jpg", match)
    except:
        print("skip")
        continue

#scores_df = pd.DataFrame(scores, index=names)
#scores_df.to_csv('olya_metrics.csv')
cv2.waitKey(0)
cv2.destroyAllWindows()

