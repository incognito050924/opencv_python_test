from skimage import transform
from skimage import filters
import cv2
import numpy as np


img = cv2.imread('images/person.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

magnitude = filters.sobel(gray.astype('float'))

for numSeams in range(20, 140, 20):
    # carved = transform.seam_carve(img, magnitude, 'horizontal', numSeams)
    carved = transform.seam_carve(img, magnitude, 'vertical', numSeams, border=80)
    print("[INFO] removing {} seams; new size: "
		"w={}, h={}".format(numSeams, carved.shape[1],
			carved.shape[0]))

    cv2.imshow('Carved', carved)
    cv2.imshow('Origin', img)
    cv2.waitKey()
    
#cv2.imshow('Origin', img)
#cv2.waitKey()
cv2.destroyAllWindows()