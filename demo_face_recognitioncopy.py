import cv2
import time
import numpy as np
from detection.FaceDetector import FaceDetector
from recognition.FaceRecognition import FaceRecognition
from classifier.FaceClassifier import FaceClassifier

# Khởi tạo các đối tượng
face_detector = FaceDetector()
face_recognition = FaceRecognition()
face_classifier = FaceClassifier('./classifier/trained_classifier.pkl')

# Đọc ảnh từ file
image_path = r'C:\Dataset\FaceRecognition\Original Images\Original Images\Natalie Portman\Natalie Portman_0.jpg'  # Thay đổi thành đường dẫn đến ảnh của bạn
frame = cv2.imread(image_path)

# Kiểm tra nếu ảnh được tải thành công
if frame is None:
    print("Error: Không thể tải ảnh.")
else:
    print('Start Recognition!')

    # Resize ảnh (tùy chọn)
    frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)

    # Duyệt qua các khuôn mặt phát hiện
    find_results = []
    frame = frame[:, :, 0:3]  # Chuyển đổi ảnh sang 3 kênh (BGR)
    boxes, scores = face_detector.detect(frame)

    # Lọc các khuôn mặt có điểm số > 0.3
    face_boxes = boxes[np.argwhere(scores>0.3).reshape(-1)]
    face_scores = scores[np.argwhere(scores>0.3).reshape(-1)]
    print('Detected_FaceNum: %d' % len(face_boxes))

    if len(face_boxes) > 0:
        for i in range(len(face_boxes)):
            box = face_boxes[i]
            cropped_face = frame[box[0]:box[2], box[1]:box[3], :]
            cropped_face = cv2.resize(cropped_face, (160, 160), interpolation=cv2.INTER_AREA)
            feature = face_recognition.recognize(cropped_face)
            name = face_classifier.classify(feature)

            # Vẽ hình chữ nhật bao quanh khuôn mặt
            cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)

            # Hiển thị tên khuôn mặt dưới khuôn mặt
            text_x = box[1]
            text_y = box[2] + 20
            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 0, 255), thickness=1, lineType=2)
    else:
        print('Unable to align')

    # Tính toán FPS (cải tiến)
    curTime = time.time()
    # Bạn có thể bỏ qua tính FPS cho ảnh đơn, hoặc tính một cách đơn giản
    fps = 1 / (curTime - time.time() + 0.0001)  # Thêm một số nhỏ để tránh chia cho 0
    str_fps = 'FPS: %2.3f' % fps
    text_fps_x = len(frame[0]) - 150
    text_fps_y = 20
    cv2.putText(frame, str_fps, (text_fps_x, text_fps_y),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)

    # Hiển thị ảnh với khuôn mặt đã nhận diện
    cv2.imshow('Face Recognition', frame)

    # Lưu ảnh kết quả nếu cần
    cv2.imwrite('recognized_face_result.jpg', frame)

    # Chờ để đóng cửa sổ ảnh khi nhấn phím bất kỳ
    cv2.waitKey(0)
    cv2.destroyAllWindows()
