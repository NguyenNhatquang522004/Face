from flask import Flask, request, jsonify 
import requests
from flask_cors import CORS
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import os
import sys
import shutil
sys.path.append(os.path.abspath("C:/Dataset/FaceRecognition"))
from recognition.facenet import get_dataset
from recognition.FaceRecognition import FaceRecognition
from detection.FaceDetector import FaceDetector
from classifier.FaceClassifier import FaceClassifier
from collections import Counter
app = Flask(__name__)
CORS(app)
face_detector = FaceDetector()
face_recognition = FaceRecognition()
face_classfier = FaceClassifier()
# base_folder = r'C:\Dataset\FaceRecognition\client'
base_folder = r'C:\Dataset\FaceRecognition\client'
OUTPUT_MODEL = r'C:\Dataset\FaceRecognition\classifier\trained_classifier.pkl'
def adjust_gamma(image, gamma=1.0):
    # Tạo bảng tra cứu gamma
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # Áp dụng gamma correction
    return cv2.LUT(image, table)
def delete_label_folder(label):
    # Đường dẫn tới thư mục cần xóa
    label_folder = os.path.join(base_folder, label)

    # Kiểm tra nếu thư mục tồn tại và tiến hành xóa
    if os.path.isdir(label_folder):
        shutil.rmtree(label_folder)
        print(f"Đã xóa thư mục: {label_folder}")
    else:
        print(f"Thư mục {label_folder} không tồn tại.")

        
def decode_and_save_images(images_base64, label):
    label_folder = os.path.join(base_folder, label)
    os.makedirs(label_folder, exist_ok=True)
    
    for idx, img_base64 in enumerate(images_base64):
        img_base64 = img_base64.replace("data:image/jpeg;base64,", "")
        image_data = base64.b64decode(img_base64)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gamma = 1.5  # Tăng giá trị gamma để làm sáng vùng tối
        brightened_image = adjust_gamma(img, gamma=gamma)
        # Lưu hình ảnh vào thư mục với tên bao gồm nhãn và số thứ tự
        img_path = os.path.join(label_folder, f'{label}_{idx}.jpg')
        cv2.imwrite(img_path, brightened_image)
        
    
def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        print(i)
        image_paths_flat += dataset[i].image_paths
        labels_flat += [dataset[i].name] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat
def adjust_gamma(image, gamma=1.0):
    # Tạo bảng tra cứu gamma
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # Áp dụng gamma correction
    return cv2.LUT(image, table)
@app.route("/Training", methods=['POST'])
def Training():
    if 'file' not in request.json or 'label' not in request.json:
        return jsonify({'message': 'No file or label provided'}), 400
    try:
        
        files = request.json['file']
        label = request.json['label']
        delete_label_folder(label)
        decode_and_save_images(files,label)
        dataset = get_dataset(base_folder)
        print(dataset)
        paths, labels = get_image_paths_and_labels(dataset)
        print(paths)
        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        image_size = 160
        nrof_images = len(paths)
        features = np.zeros((2*nrof_images, 128))
        labels = np.asarray(labels).repeat(2)
        for i in range(nrof_images):
            img = cv2.imread(paths[i])
            if img is None:
                print('Open image file failed: ' + paths[i])
                continue
            boxes, scores = face_detector.detect(img)
            if len(boxes) < 0 or scores[0] < 0.5:
                print('No face found in ' + paths[i])
                continue
            else:
                print("yes")

            cropped_face = img[boxes[0][0]:boxes[0][2], boxes[0][1]:boxes[0][3], :]
            cropped_face_flip = cv2.flip(cropped_face,1)
            features[2*i,:] = face_recognition.recognize(cropped_face)
            features[2*i+1,:] = face_recognition.recognize(cropped_face_flip)
        print('Start training for images')
        face_classfier.train(features, labels, model='svm', save_model_path=OUTPUT_MODEL)
        delete_label_folder(label)           
        return jsonify({
            'message': 'Images processed successfully',
            'label': label,
        }), 200
    except Exception as e:
        delete_label_folder(label) 
        return jsonify({'message': f'Error processing images: {str(e)}'}), 400

    
@app.route("/Check", methods=['POST'])
def recognize_faces():
    """
    Nhận diện khuôn mặt từ base64 được gửi lên, chỉ thành công nếu `label` khớp với `name`.
    """
    try:
        if 'file' not in request.json or 'label' not in request.json:
            return jsonify({"message": "No file or label provided"}), 400

        files = request.json['file']
        label = request.json['label']
        results = []
        detected_count = 0  # Số lượng ảnh nhận diện được khuôn mặt
        matching_count = 0  # Số lượng khuôn mặt khớp với nhãn
        total_images = len(files)  # Tổng số ảnh gửi lên

        for idx, base64_image in enumerate(files):
            # Giải mã ảnh từ base64
            img_base64 = base64_image.replace("data:image/jpeg;base64,", "")
            image_data = base64.b64decode(img_base64)
            image = Image.open(BytesIO(image_data)).convert('RGB')
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Làm sáng ảnh bằng gamma correction
            gamma = 1.5  # Tăng giá trị gamma để làm sáng vùng tối
            brightened_image = adjust_gamma(img, gamma=gamma)

            # Phát hiện khuôn mặt
            boxes, scores = face_detector.detect(brightened_image)
            face_boxes = boxes[np.argwhere(scores > 0.3).reshape(-1)]
            face_scores = scores[np.argwhere(scores > 0.3).reshape(-1)]

            if len(face_boxes) > 0:
                detected_count += 1  # Tăng đếm nếu phát hiện được khuôn mặt
                for i in range(len(face_boxes)):
                    box = face_boxes[i]
                    cropped_face = brightened_image[box[0]:box[2], box[1]:box[3], :]
                    cropped_face = cv2.resize(cropped_face, (160, 160), interpolation=cv2.INTER_AREA)

                    # Trích xuất đặc trưng và phân loại
                    feature = face_recognition.recognize(cropped_face)
                    name = face_classfier.classify(feature)

                    # Kiểm tra nếu tên khớp với nhãn
                    if name == label:
                        matching_count += 1

                    results.append({
                        "index": idx,
                        "name": name,
                        "box": box.tolist(),
                        "matched": name == label
                    })
            else:
                results.append({
                    "index": idx,
                    "name": "Unknown",
                    "box": None,
                    "matched": False
                })

        # Tính tỷ lệ nhận diện khuôn mặt và tỷ lệ khớp nhãn
        detection_rate = (detected_count / total_images) * 100
        matching_rate = (matching_count / total_images) * 100

        # Kiểm tra điều kiện thành công: phải khớp nhãn >= 80%
        success = matching_rate >= 80

        return jsonify({
            "message": "Face recognition completed",
            "results": results,
            "detection_rate": detection_rate,
            "matching_rate": matching_rate,
            "success": success
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500
@app.route("/Detect_Tranining", methods=['POST'])
def detect():
    try:
        files = request.json['file']
        label = request.json['label']
        results = []
        detected_count = 0  # Đếm số lượng ảnh phát hiện được khuôn mặt
        total_images = len(files)  # Tổng số ảnh gửi lên
        detected_images = []

        for idx, img_base64 in enumerate(files):
            # Giải mã base64
            img_base64 = img_base64.replace("data:image/jpeg;base64,", "")
            image_data = base64.b64decode(img_base64)
            image = Image.open(BytesIO(image_data)).convert('RGB')
            img = np.array(image)  # Chuyển sang numpy array
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Làm sáng ảnh bằng gamma correction
            gamma = 1.5  # Tăng giá trị gamma để làm sáng vùng tối
            brightened_image = adjust_gamma(img, gamma=gamma)

            # Phát hiện khuôn mặt
            boxes, scores = face_detector.detect(brightened_image)
            if len(boxes) == 0 or scores[0] < 0.5:
                results.append({'index': idx, 'status': 'no face detected'})
            else:
                print("yes")
                detected_count += 1  # Tăng bộ đếm nếu phát hiện được khuôn mặt
                results.append({'index': idx, 'status': 'face detected'})

                # Thêm ảnh đã phát hiện thành công vào danh sách
                detected_images.append("data:image/jpeg;base64," + base64.b64encode(cv2.imencode('.jpg', brightened_image)[1]).decode('utf-8'))

        # Tính tỷ lệ phát hiện khuôn mặt
        detection_rate = (detected_count / total_images) * 100

        # Kiểm tra tỷ lệ phát hiện
        success = detection_rate >= 80  # Thành công nếu phát hiện >= 80%

        if success:
            # Gửi HTTP request tới route /Training
            training_response = requests.post(
                url="http://localhost:5000/Training",  # URL tới route /Training
                json={
                    "file": files,
                    "label": label
                }
            )

            # Xử lý kết quả trả về từ /Training
            if training_response.status_code == 200:
                training_result = training_response.json()
                print("Training thành công:", training_result)
            else:
                print("Lỗi trong quá trình Training:", training_response.json())

        return jsonify({
            'message': 'Processed images successfully',
            'results': results,
            'detection_rate': detection_rate,
            'success': success
        }), 200

    except Exception as e:
        return jsonify({'message': f'Error processing images: {str(e)}'}), 500
    
@app.route("/Detect_Check", methods=['POST'])
def detect_check():
    try:
        files = request.json['file']
        label = request.json['label']
        results = []
        detected_count = 0  # Đếm số lượng ảnh phát hiện được khuôn mặt
        total_images = len(files)  # Tổng số ảnh gửi lên
        detected_images = []

        for idx, img_base64 in enumerate(files):
            # Giải mã base64
            img_base64 = img_base64.replace("data:image/jpeg;base64,", "")
            image_data = base64.b64decode(img_base64)
            image = Image.open(BytesIO(image_data)).convert('RGB')
            img = np.array(image)  # Chuyển sang numpy array
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Làm sáng ảnh bằng gamma correction
            gamma = 1.5  # Tăng giá trị gamma để làm sáng vùng tối
            brightened_image = adjust_gamma(img, gamma=gamma)

            # Phát hiện khuôn mặt
            boxes, scores = face_detector.detect(brightened_image)
            if len(boxes) == 0 or scores[0] < 0.5:
                results.append({'index': idx, 'status': 'no face detected'})
            else:
                print("yes")
                detected_count += 1  # Tăng bộ đếm nếu phát hiện được khuôn mặt
                results.append({'index': idx, 'status': 'face detected'})

                # Thêm ảnh đã phát hiện thành công vào danh sách
                detected_images.append("data:image/jpeg;base64," + base64.b64encode(cv2.imencode('.jpg', brightened_image)[1]).decode('utf-8'))

        # Tính tỷ lệ phát hiện khuôn mặt
        detection_rate = (detected_count / total_images) * 100

        # Kiểm tra tỷ lệ phát hiện
        success = detection_rate >= 80  # Thành công nếu phát hiện >= 80%

        if success:
            # Gửi HTTP request tới route /Training
            training_response = requests.post(
                url="http://localhost:5000/Check",  # URL tới route /Training
                json={
                    "file": files,
                    "label": label
                }
            )
        
            
            # Xử lý kết quả trả về từ /Training
            if training_response.status_code == 200:
                training_result = training_response.json()
                print("Training thành công:", training_result)
            else:
                print("Lỗi trong quá trình Check:", training_response.json())

        return jsonify({
            'message': 'Processed images successfully',
            'results': results,
            'detection_rate': detection_rate,
            'success': success
        }), 200

    except Exception as e:
        return jsonify({'message': f'Error processing images: {str(e)}'}), 500



if __name__ == "__main__":
    app.run(debug=True)


