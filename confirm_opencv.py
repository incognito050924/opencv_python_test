import cv2

img = cv2.imread('images/person.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
# print(sobel_x)
abs_sobel_x = cv2.convertScaleAbs(sobel_x)
abs_sobel_y = cv2.convertScaleAbs(sobel_y)

energy_matrix = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

cv2.imshow('Origin', img)
cv2.imshow('Abs Sobel X', abs_sobel_x)
cv2.imshow('Abs Sobel Y', abs_sobel_y)
cv2.imshow('Energy Matrix', energy_matrix)
cv2.waitKey()
cv2.destroyAllWindows()
