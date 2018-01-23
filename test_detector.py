import cv2
import detector
import skin_analyzer


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

d = detector.LandmarkDetector()
# landmarks_img = d.annotate_landmarks(img, d.get_landmarks(img))
# cv2.imshow('img', landmarks_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

features, points = d.detect_facial_feature(img, visible=False)

# print('right_eye_side', d.range_from_landmark(features['right_eye_side']))

#
# for key, value in features.items():
#     cv2.imshow(key, value)
# cv2.waitKey()
# cv2.destroyAllWindows()

# img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 200)
# cv2.imshow('Gray', gray)
# cv2.imshow('Edges', edges)
# cv2.waitKey()
# cv2.destroyAllWindows()

extractor = skin_analyzer.Extractor()
extractor.extract_pore(features['pore_roi'])
extractor.extract_pigmentation(features['skin_roi'])
extractor.extract_wrinkle(features['wrinkle_roi'])
extractor.extract_erythema(features['skin_roi'])
print(extractor.pore, extractor.pigmentation, extractor.wrinkle, extractor.erythema)
analyzer = skin_analyzer.Analyzer()