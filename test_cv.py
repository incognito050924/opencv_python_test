import cv2
import numpy as np
from PIL import Image
import io

im = Image.open('images/bang_hair.jpg')
buffer = io.BytesIO()
im.save(buffer, format='JPEG')
print(type(buffer.getvalue()))


np_arr = np.fromstring(buffer.getvalue(), np.uint8)
img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
print(img.shape)

cv2.imshow('asdasdasd', img)
cv2.waitKey()
cv2.destroyAllWindows()