import cv2
import os
import numpy as np
from recognition.facenet import get_dataset
from recognition.FaceRecognition import FaceRecognition
from detection.FaceDetector import FaceDetector
from classifier.FaceClassifier import FaceClassifier
face_detector = FaceDetector()
face_recognition = FaceRecognition()
face_classfier = FaceClassifier()
def adjust_gamma(image, gamma=1.0):
    # Tạo bảng tra cứu gamma
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # Áp dụng gamma correction
    return cv2.LUT(image, table)

# Đọc ảnh từ file
image = cv2.imread(r'C:\Dataset\FaceRecognition\down.jpg')

# Kiểm tra nếu ảnh được tải
if image is None:
    print("Không thể tải ảnh. Kiểm tra lại đường dẫn.")
    exit()

# Điều chỉnh gamma (giá trị <1 để làm sáng vùng tối)
gamma = 1.5  # Giá trị gamma, tăng để làm sáng vùng tối
brightened_image = adjust_gamma(image, gamma=gamma)
boxes, scores = face_detector.detect(image)
if len(boxes) < 0 or scores[0] < 0.5:
  print('No face found in ' )
else:
  print("te")

# Hiển thị ảnh gốc và ảnh đã xử lý
cv2.imshow('Ảnh gốc', image)
cv2.imshow('Ảnh làm sáng vùng tối', brightened_image)

# Lưu ảnh kết quả (tùy chọn)
cv2.imwrite('brightened_dark_regions.jpg', brightened_image)

# Đợi và đóng các cửa sổ
cv2.waitKey(0)
cv2.destroyAllWindows()
