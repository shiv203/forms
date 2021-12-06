import cv2
import numpy as np
from form_digitization import bounding_box

def draw_rectangle(image : np.array, bbox : list, color : tuple) -> np.array:
    """ draw a rectangle """
    image = np.asarray(image, np.uint8)
    image_crop = image[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
    white_rect = np.ones(image_crop.shape, dtype=np.uint8) * color
    #print(white_rect.shape, image_crop.shape)
    res = image_crop * 0.7 + white_rect * 0.3
    image[bbox[1]:bbox[3],bbox[0]:bbox[2]] = res
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 5)
    return image

def plot_line_of_sight(process, image : np.array) -> np.array:
    """ plot all the right and bottom line of sight elements """
    image = generate_output(image, process)
    line_of_sight  = process.line_of_sight
    #print(line_of_sight)
    for key_i in line_of_sight:
        #print(line_of_sight[key_i])
        bb =  process.document_features.document_elements[key_i]
        right_id = line_of_sight[key_i]["right"]
        bottom_id = line_of_sight[key_i]["bottom"]
        if right_id is not None:
            right = process.document_features.document_elements[right_id]
            cv2.circle(image, ( int((bb.x1 + bb.x2)/2), int((bb.y1 + bb.y2)/2) ), 25, (122, 0, 122), -1)
            cv2.circle(image, (int((right.x1 + right.x2)/2), int((right.y1 + right.y2)/2)), 25, (122, 0, 122), -1)
            cv2.line(image, ( int((bb.x1 + bb.x2)/2), int((bb.y1 + bb.y2)/2) ), ( int((right.x1 + right.x2)/2), int((right.y1 + right.y2)/2) ), (0, 255, 0), 4)
        if bottom_id is not None:
            bottom = process.document_features.document_elements[bottom_id]
            cv2.circle(image, (bb.x2, int((bb.y1 + bb.y2)/2)), 10, (0, 122, 122), -1)
            cv2.circle(image, (bottom.x1, int((bottom.y1 + bottom.y2)/2)), 10, (0, 122, 122), -1)
            #cv2.line(image, ( int((bb.x1 + bb.x2)/2), int((bb.y1 + bb.y2)/2) ), ( int((bottom.x1 + bottom.x2)/2), int((bottom.y1 + bottom.y2)/2) ), (0, 0, 255), 4)
            cv2.line(image, (bb.x2, int((bb.y1 + bb.y2)/2)), (bottom.x1, int((bottom.y1 + bottom.y2)/2)), (0, 0, 255), 5)
    return image




def add_neighbors(process, image : np.array, id: int, number_of_neigbors: int) -> np.array:
    """ given an form field element collect all its links """
    form_element_box = process.document_features.document_elements[id]
    x1, y1, x2, y2 = form_element_box.x1, form_element_box.y1, form_element_box.x2, form_element_box.y2
    neigbors = process.adjacency_list[id]
    xmin, ymin, xmax, ymax = x1, y1, x2, y2
    for n in neigbors[:number_of_neigbors]:
        xmin = min(xmin, process.document_features.document_elements[n].x1)
        ymin = min(ymin, process.document_features.document_elements[n].y1)
        xmax = max(xmax, process.document_features.document_elements[n].x2)
        ymax = max(ymax, process.document_features.document_elements[n].y2)
    
    list_of_images = []
    for i in range(number_of_neigbors):
        new_image = image.copy()
        text_element_box =  process.document_features.document_elements[process.adjacency_list[id][i]]
        tx1, ty1, tx2, ty2 = text_element_box.x1, text_element_box.y1, text_element_box.x2, text_element_box.y2
        cv2.line(new_image, (x1, y1), (tx1, ty1), (0, 255, 0), 10 )
        cv2.putText(new_image, str(i+1), (tx1, ty2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, 3)
        new_image = new_image[ymin:ymax, xmin:xmax, :]
        list_of_images.append(new_image)
    return list_of_images

def generate_output(image, process):
    """ generate display image outputs ... """
    bounding_box = process.document_features.document_elements
    for k in bounding_box:
        ## text element
        if bounding_box[k].label == 3:
            color = (0, 122, 122)
            bbox = [bounding_box[k].x1, bounding_box[k].y1, bounding_box[k].x2, bounding_box[k].y2 ]
            image = draw_rectangle(image, bbox, color)
        else:
            color = (122, 0, 122)
            bbox = [bounding_box[k].x1, bounding_box[k].y1, bounding_box[k].x2, bounding_box[k].y2 ]
            image = draw_rectangle(image, bbox, color)
    return image