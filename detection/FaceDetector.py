import os
import time
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(__file__) + '/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'protos/face_label_map.pbtxt'

class FaceDetector:
    def __init__(self):
        # Load the detection model
        self.detection_graph = tf.compat.v1.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(BASE_DIR + PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Initialize tensors
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Create a session in TF 2.x compatible mode
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

    def __del__(self):
        self.sess.close()

    def detect(self, image):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_expanded = np.expand_dims(image, axis=0)

        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [self.boxes, self.scores, self.classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})
        elapsed_time = time.time() - start_time
        print('Inference time cost: {}'.format(elapsed_time))

        # Scale boxes to real position
        boxes[0, :, [0, 2]] = boxes[0, :, [0, 2]] * image.shape[0]
        boxes[0, :, [1, 3]] = boxes[0, :, [1, 3]] * image.shape[1]
        return np.squeeze(boxes).astype(int), np.squeeze(scores)
