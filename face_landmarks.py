import dlib
import cv2
import numpy as np
import math

PREDICTIOR_PATH = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTIOR_PATH)
detector = dlib.get_frontal_face_detector()

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise  TooManyFaces
    if len(rects) < 1:
        raise NoFaces

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0,0,255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
        #print(idx, pos)
    return im

def eye_dilate(eyes, x_rate=1.6, y_rate=2):
    # dilate_x = abs(int((eyes[3][0, 0] - eyes[0][0, 0]) / 2 * x_rate))
    # dilate_y = abs(int((eyes[5][0, 1] - eyes[1][0, 1]) / 2 * y_rate))
    #
    # eye_weights = [[-dilate_x, 0], [int(-dilate_x / 2), -dilate_y], [int(dilate_x / 2), -dilate_y],
    #                [dilate_x, 0], [int(dilate_x / 2), dilate_y], [int(-dilate_x / 2), dilate_y]]
    # return eyes + eye_weights
    result = eyes

    x_half_length = math.ceil((eyes[3][0, 0] - eyes[0][0, 0]) / 2)
    x_center = eyes[3][0, 0] - x_half_length

    x2_half_length = math.ceil((eyes[2][0, 0] - eyes[1][0, 0]) / 2)
    x2_center = eyes[2][0, 0] - x2_half_length

    x3_half_length = math.ceil((eyes[4][0, 0] - eyes[5][0, 0]) / 2)
    x3_center = eyes[4][0, 0] - x3_half_length

    y1_half_length = math.ceil((eyes[5][0, 1] - eyes[1][0, 1]) / 2)
    y1_center = eyes[1][0, 1] + y1_half_length

    y2_half_length = math.ceil((eyes[4][0, 1] - eyes[2][0, 1]) / 2)
    y2_center = eyes[2][0, 1] + y2_half_length

    result[0][0, 0] = x_center - x_half_length * x_rate
    result[1][0, 0] = x2_center - x2_half_length * x_rate
    result[1][0, 1] = y1_center - y1_half_length * y_rate
    result[2][0, 0] = x2_center + x2_half_length * x_rate
    result[2][0, 1] = y2_center - y2_half_length * y_rate
    result[3][0, 0] = x_center + x_half_length * x_rate
    result[4][0, 0] = x3_center + x3_half_length * x_rate
    result[4][0, 1] = y2_center + y2_half_length * y_rate
    result[5][0, 0] = x3_center - x3_half_length * x_rate
    result[5][0, 1] = y1_center + y1_half_length * y_rate

    return result

def eye_brow_dilate(eye_brows, x_rate=1.1, y_rate=1.3):
    result = eye_brows

    x_half_length = math.ceil((eye_brows[4][0, 0] - eye_brows[0][0, 0]) / 2)
    x_center = eye_brows[4][0, 0] - x_half_length

    x2_half_length = math.ceil((eye_brows[3][0, 0] - eye_brows[1][0, 0]) / 2)
    x2_center = eye_brows[3][0, 0] - x2_half_length

    y_low = eye_brows[0][0, 1] if eye_brows[0][0, 1] > eye_brows[4][0, 1] else eye_brows[4][0, 1]
    y1_length = y_low - eye_brows[1][0, 1]
    y2_length = y_low - eye_brows[2][0, 1]
    y3_length = y_low - eye_brows[3][0, 1]

    result[0][0, 0] = x_center - x_half_length *  x_rate
    result[1][0, 0] = x2_center - x2_half_length * x_rate
    result[1][0, 1] = y_low - y1_length * y_rate
    result[2][0, 1] = y_low - y2_length * y_rate
    result[3][0, 0] = x2_center + x2_half_length * x_rate
    result[3][0, 1] = y_low - y3_length * y_rate
    result[4][0, 0] = x_center + x_half_length * x_rate

    return result

def nose_dilate(nose, x_rate=1.3, y_rate=1):
    result = nose

    x1_half_length = math.ceil((nose[8][0, 0] - nose[4][0, 0]) / 2)
    x1_center = nose[8][0, 0] - x1_half_length

    x2_half_length = math.ceil((nose[7][0, 0] - nose[5][0, 0]) / 2)
    x2_center = nose[7][0, 0] - x2_half_length

    y_top = nose[0][0, 1]
    y1_length = nose[4][0, 1] - y_top
    y2_length = nose[5][0, 1] - y_top
    y3_length = nose[6][0, 1] - y_top
    y4_length = nose[7][0, 1] - y_top
    y5_length = nose[8][0, 1] - y_top

    result[4][0, 0] = x1_center - x1_half_length * x_rate
    result[4][0, 1] = y_top + y1_length * y_rate
    result[5][0, 0] = x2_center - x2_half_length * x_rate
    result[5][0, 1] = y_top + y2_length * y_rate
    result[6][0, 1] = y_top + y3_length * y_rate
    result[7][0, 0] = x2_center + x2_half_length * x_rate
    result[7][0, 1] = y_top + y4_length * y_rate
    result[8][0, 0] = x1_center + x1_half_length * x_rate
    result[8][0, 1] = y_top + y5_length * y_rate

    return result

def mouth_dilate(mouth, x_rate=1.2, y_rate=1.2):
    result = mouth

    x1_half_length = math.ceil((mouth[6][0, 0] - mouth[0][0, 0]) / 2)
    x1_center = mouth[6][0, 0] - x1_half_length

    x2_half_length = math.ceil((mouth[5][0, 0] - mouth[1][0, 0]) / 2)
    x2_center = mouth[5][0, 0] - x2_half_length

    x3_half_length = math.ceil((mouth[4][0, 0] - mouth[2][0, 0]) / 2)
    x3_center = mouth[4][0, 0] - x3_half_length

    x4_half_length = math.ceil((mouth[7][0, 0] - mouth[11][0, 0]) / 2)
    x4_center = mouth[7][0, 0] - x4_half_length

    x5_half_length = math.ceil((mouth[8][0, 0] - mouth[10][0, 0]) / 2)
    x5_center = mouth[8][0, 0] - x5_half_length

    y1_half_length = math.ceil((mouth[11][0, 1] - mouth[1][0, 1]) / 2)
    y1_center = mouth[1][0, 1] + y1_half_length

    y2_half_length = math.ceil((mouth[10][0, 1] - mouth[2][0, 1]) / 2)
    y2_center = mouth[2][0, 1] + y2_half_length

    y3_half_length = math.ceil((mouth[9][0, 1] - mouth[3][0, 1]) / 2)
    y3_center = mouth[3][0, 1] + y3_half_length

    y4_half_length = math.ceil((mouth[8][0, 1] - mouth[4][0, 1]) / 2)
    y4_center = mouth[4][0, 1] + y4_half_length

    y5_half_length = math.ceil((mouth[7][0, 1] - mouth[5][0, 1]) / 2)
    y5_center = mouth[5][0, 1] + y5_half_length


    result[0][0, 0] = x1_center - x1_half_length * x_rate
    result[1][0, 0] = x2_center - x2_half_length * x_rate
    result[1][0, 1] = y1_center - y1_half_length * y_rate
    result[2][0, 0] = x3_center - x3_half_length * x_rate
    result[2][0, 1] = y2_center - y2_half_length * y_rate
    result[3][0, 1] = y3_center - y3_half_length * y_rate
    result[4][0, 0] = x3_center + x3_half_length * x_rate
    result[4][0, 1] = y4_center - y4_half_length * y_rate
    result[5][0, 0] = x2_center + x2_half_length * x_rate
    result[5][0, 1] = y5_center - y5_half_length * y_rate
    result[6][0, 0] = x1_center + x1_half_length * x_rate
    result[7][0, 0] = x4_center + x4_half_length * x_rate
    result[7][0, 1] = y5_center + y5_half_length * y_rate
    result[8][0, 0] = x5_center + x5_half_length * x_rate
    result[8][0, 1] = y4_center + y4_half_length * y_rate
    result[9][0, 1] = y3_center + y3_half_length * y_rate
    result[10][0, 0] = x5_center - x5_half_length * x_rate
    result[10][0, 1] = y2_center + y2_half_length * y_rate
    result[11][0, 0] = x4_center - x4_half_length * x_rate
    result[11][0, 1] = y1_center + y1_half_length * y_rate

    return result

def get_landmark_dict(landmarks):
    featuers = {}
    featuers['jaw'] = landmarks[0:17]
    featuers['right_eyebrow'] = landmarks[17:22]
    featuers['left_eyebrow'] = landmarks[22:27]
    featuers['nose'] = landmarks[27:36]
    featuers['right_eye'] = landmarks[36:42]
    featuers['left_eye'] = landmarks[42:48]
    featuers['mouth'] = landmarks[48:68]
    return featuers

def draw_borderline(im, landmarks, colors=None, alpha=0.75,
                    use_feature=['jaw', 'right_eyebrow', 'left_eyebrow', 'nose', 'right_eye', 'left_eye', 'mouth']):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = im.copy()
    output = im.copy()

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    facial_landmarks = get_landmark_dict(landmarks)
    for i, feature in enumerate(use_feature):
        pts =  facial_landmarks[feature]
        if feature == 'jaw':
            for idx in range(1, len(pts)):
                ptA = (pts[idx - 1][0, 0], pts[idx - 1][0, 1])
                ptB = (pts[idx][0, 0], pts[idx][0, 1])
                cv2.line(overlay, ptA, ptB, colors[i], 2)
                cv2.circle(overlay, ptB, 3, color=(0, 255, 255))

        else:
            if feature == 'right_eye' or feature == 'left_eye':
                pts = eye_dilate(facial_landmarks[feature])
            elif feature == 'right_eyebrow' or feature == 'left_eyebrow':
                pts = eye_brow_dilate(facial_landmarks[feature])
            elif feature == 'mouth':
                pts = mouth_dilate(facial_landmarks[feature])
            elif feature =='nose':
                pts = nose_dilate(facial_landmarks[feature])
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

            for idx in range(len(pts)):
                #ptA = tuple(pts[idx - 1])
                #ptB = tuple(pts[idx])
                pos = (pts[idx][0, 0], pts[idx][0, 1])
                cv2.circle(overlay, pos, 3, color=(0, 255, 255))

    cv2.imshow('Overlay_Landmarks', overlay)


img = cv2.imread('images/bang_hair.jpg')
landmarks = get_landmarks(img)
# print(landmarks)
image_with_landmarks = annotate_landmarks(img, landmarks=landmarks)
draw_borderline(img, landmarks)

cv2.imshow('Image with Landmarks', image_with_landmarks)
cv2.waitKey()
cv2.destroyAllWindows()

# print(eye_dilate(get_landmark_dict(landmarks)))