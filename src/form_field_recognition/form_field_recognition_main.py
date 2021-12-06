import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
sys.path.append("./form_field_recognition/")

import label_map_util



from configparser import SafeConfigParser

parser = SafeConfigParser()
parser.read("../product-package/conf/server.conf")
model_path = parser.get("form_field_recognition", "form_field_recognition_model")
text_path = parser.get("form_field_recognition", "form_field_recognition_text")

# post processing functions
def inter_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    epsilon = 10**-20
    
    interArea = max(0, xB - xA + epsilon) * max(0, yB - yA + epsilon)
    boxAArea = (boxA[2] - boxA[0] + epsilon) * (boxA[3] - boxA[1] + epsilon)
    boxBArea = (boxB[2] - boxB[0] + epsilon) * (boxB[3] - boxB[1] + epsilon)
    iou = (interArea / float(boxAArea + boxBArea - interArea))*100.0
    return iou

def post_process( boxes, scores, classes):
    predicted_boxes = []
    y_min, x_min, y_max, x_max = boxes.T
    for cls, xmin, ymin, xmax, ymax, score in zip(classes-1, x_min, y_min, x_max, y_max, scores):
        if cls==2:
            predicted_boxes.append([cls, score, xmin, ymin-0.02, xmax, ymax])
        else:
            predicted_boxes.append([cls, score, xmin, ymin, xmax, ymax])
    
    remove_ele_ar = []
    for i in range(len(predicted_boxes)):
        boxA = predicted_boxes[i]
        for j in range(len(predicted_boxes)):
            boxB = predicted_boxes[j]
            IoU = inter_over_union(boxA[2:], boxB[2:])

            if IoU >= 50 and IoU != 100:
                if boxA[1] > boxB[1]:
                    remove_ele_ar.append(boxB)
                else:
                    remove_ele_ar.append(boxA)
    
    if remove_ele_ar != []:
        for ele in remove_ele_ar:
            try:
                predicted_boxes.remove(ele)
            except ValueError:
                continue
    try:
        predicted_boxes = np.array(predicted_boxes)
        predicted_boxes[:,3][predicted_boxes[:,0] == 2] += 0.02
        cls = np.array(np.matrix(predicted_boxes).T[0])[0]
        box = np.array(np.matrix(predicted_boxes).T[2:].T)
        return cls, box
    except IndexError:
        return np.ndarray(shape=(0,), dtype=float), np.ndarray(shape=(0,4), dtype=float)

def save_and_post_process( boxes, scores, classes, num_detection):
    boxes, scores, classes = boxes[0][:int(num_detection[0])], scores[0][:int(num_detection[0])], classes[0][:int(num_detection[0])]
    
    if boxes[:int(num_detection)].shape[0]!=0:
        classes, boxes = post_process( boxes, scores, classes)
    
    result_values = []
    for i,j in zip(boxes, classes):
        xmin, ymin, xmax, ymax = i
        
        width = xmax-xmin
        height = ymax-ymin
        mid_x = xmin+(width/2)
        mid_y = ymin+(height/2)
        
        result_values.append([int(j), xmin, ymin, xmax, ymax])

    return result_values



class form_field_recognition:
    """ form field recognition """

    def __init__(self, ):
        self.PATH_TO_CKPT = model_path
        self.PATH_TO_LABELS = text_path

        self.NUM_CLASSES = 3

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')

        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)

        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES)
        self.category_index = label_map_util.create_category_index(self.categories)


    def recognize(self, image):
        output = []
        with self.detection_graph.as_default():
            with tf.compat.v1.Session(graph=self.detection_graph) as sess:
            
                
                image_np_expanded = np.expand_dims(image, axis=0)
                
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detection = self.detection_graph.get_tensor_by_name('num_detections:0')

                start = time.time()
                
                (boxes, scores, classes, num_detection) = sess.run([boxes, scores, classes, num_detection], 
                                                                feed_dict = {image_tensor:image_np_expanded})

                end = time.time()
                
                print("FORM field model time : ", (end - start))
                start = time.time()
                output =  save_and_post_process( boxes, scores, classes, num_detection)
                end = time.time()
                print("post processing :: ", (end - start))

                for i in range(len(output)):
                    output[i][1] = int(output[i][1] * image.shape[1])
                    output[i][2] = int(output[i][2] * image.shape[0])
                    output[i][3] = int(output[i][3] * image.shape[1])
                    output[i][4] = int(output[i][4] * image.shape[0])
        return output