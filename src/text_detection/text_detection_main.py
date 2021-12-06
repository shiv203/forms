import tensorflow as tf
import numpy as np
import cv2
import sys
from PIL import Image
import tesserocr
api = tesserocr.PyTessBaseAPI()
sys.path.append("./text_detection/")

from text_detect_utils import getDetBoxes

from configparser import SafeConfigParser

parser = SafeConfigParser()
parser.read("../product-package/conf/server.conf")
model_path = parser.get("text_detection", "text_detection_model")




def normalize_mean_variance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape
    target_size = mag_ratio * max(height, width)
    if target_size > square_size:
        target_size = square_size
    ratio = target_size / max(height, width)
    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32
    size_heatmap = (int(target_w/2), int(target_h/2))
    return resized, ratio, size_heatmap



class text_detection:
    """text detection """

    def __init__(self, ):
        with tf.compat.v1.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        self.input_tensor = graph.get_tensor_by_name("input_tensor:0")
        self.y_tensor = graph.get_tensor_by_name("y_tensor:0")
        self.feature_tensor = graph.get_tensor_by_name("feature_tensor:0")
        self.sess = tf.compat.v1.Session(graph=graph)

    def detect(self, image):
        orginal_image = image.copy()
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_height,image_width,image_channel = image.shape
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
        w_ratio = target_ratio
        h_ratio = target_ratio
        image = normalize_mean_variance(img_resized)
        image = np.expand_dims(image, 0)
        y_val, feature_val = self.sess.run([self.y_tensor, self.feature_tensor], {self.input_tensor: image})
        score_text = y_val[0, :, :, 0]
        score_link = y_val[0, :, :, 1]
        text_threshold = 0.4
        link_threshold = 0.4
        low_text = 0.3
        boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, False)
        boxes = np.array(boxes)
        if len(boxes)>0:
            boxes_x = boxes[:, :, 0] / w_ratio * 2
            boxes_x[boxes_x<0]=0
            boxes_x[boxes_x>image_width]=image_width
            boxes_y = boxes[:, :, 1] / h_ratio * 2
            boxes_y[boxes_y<0]=0
            boxes_y[boxes_y>image_height]=image_height
            final_coordinates = np.stack([boxes_x, boxes_y], axis=2)
        else:
            final_coordinates = np.ones((0,0,2))


        out = []

        for i in range(final_coordinates.shape[0]):
            x1 = min(final_coordinates[i][0][0], final_coordinates[i][1][0], final_coordinates[i][2][0], final_coordinates[i][3][0])
            x2 = max(final_coordinates[i][0][0], final_coordinates[i][1][0], final_coordinates[i][2][0], final_coordinates[i][3][0])
            y1 = min(final_coordinates[i][0][1], final_coordinates[i][1][1], final_coordinates[i][2][1], final_coordinates[i][3][1])
            y2 = max(final_coordinates[i][0][1], final_coordinates[i][1][1], final_coordinates[i][2][1], final_coordinates[i][3][1])
            x1 = max(0, int(x1 - 5))
            y1 = max(0, int(y1 - 5))
            x2 = min(image_width, int(x2 + 5))
            y2 = min(image_height, int(y2 + 5))
            #print(x1, y1, x2, y2)
            warp = orginal_image[y1:y2 , x1:x2 , :]
            #print(image.shape, warp.shape)
            warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
            (thresh, bw_img) = cv2.threshold(warp, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img = Image.fromarray(warp)
            api.SetImage(img)
            txt = api.GetUTF8Text()
            out.append([ int(final_coordinates[i][0][0]), int(final_coordinates[i][0][1]), int(final_coordinates[i][2][0]), int(final_coordinates[i][2][1]), txt ])
        
        return out 