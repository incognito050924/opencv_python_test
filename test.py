import face_detector
import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread('images/kate-upton.jpg')
img = cv2.imread('images/images.jpg')
# img = cv2.imread('images/bang_hair.jpg')
# img = cv2.imread('images/bang_hair2.jpg')
# img = cv2.imread('images/bang_hair3.jpg') # no face
# img = cv2.imread('images/bang_hair4.jpg') # no face
# img = cv2.imread('images/bang_hair5.jpg') # no eyes
# img = cv2.imread('images/bang_hair6.jpg') # no eyes, no nose, no mouth
# img = cv2.imread('images/bang_hair7.jpg') # wrong nose
# img = cv2.imread('images/bang_hair8.jpg')
# img = cv2.imread('images/face_with_glasses.jpg') # no face
# img = cv2.imread('images/face_with_glasses2.jpg')
# img = cv2.imread('images/face_with_glasses3.jpg') # no eyes, wrong nose
# img = cv2.imread('images/face_with_glasses4.jpg') # wrong face, no eyes, wrong nose, no mouth
# img = cv2.imread('images/face_with_glasses_and_bang.jpg') # no eyes
# img = cv2.imread('images/face_with_glasses_and_bang2.jpg') # no eyes, no nose
# img = cv2.imread('images/face_with_glasses_and_bang3.jpg') # no eyes
# img = cv2.imread('images/face_with_glasses_and_beard.jpg') # no eyes
# img = cv2.imread('images/face_with_glasses_and_beard2.jpg') # no eyes

# print(img.shape)

# detector = face_detector.CascadeDetector()
# features = detector.detect_facial_feature(img, visible=True)
# avg_color = face_detector.average_color(features['right_cheek'])
#print(avg_color.shape)

# detector = face_detector.LandmarkDetector()
# landmarks = detector.get_landmarks(features['face'])
# detector.remove_landmarks_from_img(features['face'], landmarks, color=avg_color)
# detector.detect_facial_feature(img, visible=True)

detector = face_detector.CascadeDetector()
detector.test_mean(img)

# cv2.imshow('Face', features['face'])
# cv2.imshow('Eyes', features['eyes'])
# cv2.imshow('Nose', features['nose'])
# cv2.imshow('Mouth', features['mouth'])
# cv2.imshow('Right_cheek', features['right_cheek'])
# cv2.imshow('Left_cheek', features['left_cheek'])
# cv2.imshow('Skin', features['skin'])
# cv2.waitKey()
# cv2.destroyAllWindows()


# print(face_detector.get_cosine_kernel(9, 11))

# fs = 30
# f = 1
# x = np.arange(fs)
# cos_x = [ (np.cos(2*np.pi * f * (i/fs)) + 1) / 2 for i in np.arange(fs)]
# sin_x = [ (np.sin(2*np.pi * 0.5 * (i/fs)) + 0) for i in np.arange(fs)]
# plt.plot(x, sin_x)
# plt.plot(x, cos_x)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
