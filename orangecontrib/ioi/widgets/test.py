# cspace.py
import cv2 as cv
import numpy as np

from dominantcolors import find_dominant_colors

from sklearn.cluster import KMeans
from collections import Counter

def get_dominant_color(image, k=4, image_processing_size=None):

    # resize image if new dims provided
    if image_processing_size is not None:
        image = cv.resize(image, image_processing_size, interpolation=cv.INTER_AREA)

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster and assign labels to the pixels
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)

    # count labels to find most popular
    label_counts = Counter(labels)
    # print(labels)
    # print(label_counts.most_common(1))
    # print(clt.cluster_centers_)
    # subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return clt.cluster_centers_


# cap = cv2.VideoCapture()
# frame = cv.imread("/Users/jakakokosar/Pictures/Photo Booth Library/Pictures/Photo on 21-12-2019 at 20.06.jpg")
frame = cv.imread("/Users/jakakokosar/Pictures/Photo Booth Library/Pictures/Photo on 18-12-2019 at 13.33.jpg")


# Convert BGR to HSV
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

dom = get_dominant_color(hsv, k=3, image_processing_size=(10, 10))

min = np.array([10, 50, 50], np.uint8)
max = np.array([20, 255, 255], np.uint8)
print(dom)

mask_min = dom > min
mask_max = dom < max
combined_mask = mask_max & mask_min
dom[np.all(combined_mask, axis=1), :]

print(combined_mask)
print(np.sum(combined_mask, axis=1))
print(dom[np.sum(combined_mask, axis=1) == 3, :])
print(dom[np.all(combined_mask, axis=1), :])


# print(cv.cvtColor(np.uint8([[dom]]), cv.COLOR_HSV2RGB))

# # define range of orange color in HSV
# min = np.array([10, 50, 50], np.uint8)
# max = np.array([20, 255, 255], np.uint8)

# define range of blue color in HSV
# min = np.array([90,50,50])
# max = np.array([130,255,255])

# # define range of yellow color in HSV
# min = np.array([25, 50, 50], np.uint8)
# max = np.array([35, 255, 255], np.uint8)

# # define range of green color in HSV
# min = np.array([40, 50, 50], np.uint8)
# max = np.array([80, 255, 255], np.uint8)
#
# # Threshold the HSV image to get only blue colors
# mask = cv.inRange(hsv, min, max)
# # print(mask)
#
# # Bitwise-AND mask and original image
# res = cv.bitwise_and(frame,frame, mask= mask)
#
# cv.imshow('frame',frame)
# cv.imshow('mask',mask)
# cv.imshow('res',res)
# cv.waitKey(0)
# cv.destroyAllWindows()
