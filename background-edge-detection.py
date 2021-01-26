import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("utsav-womboai.jpg")
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

edges = cv.Canny(grayimg, 100, 50)
plt.subplot(121),plt.imshow(grayimg,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()