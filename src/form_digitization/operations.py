"""" combining data structure from bounding box to add heuristic functions """
import math
import sys
sys.path.append("./form_digitization/")
from bounding_box import bounding_box
from bounding_box import document_properties


def get_peripheral_points(point1, point2, num_of_points):
    """ return num_of_points on the line point1 - point2 
    number of points minimum should be 1 """
    list_of_pts = []
    for i in range(1, num_of_points):
        x = (i*point1[0] + (num_of_points - i)*point2[0]) / num_of_points
        y = (i*point1[1] + (num_of_points - i)*point2[1]) / num_of_points
        list_of_pts.append((x, y))
    return list_of_pts

def distance_between_points(pts1, pts2):
    """ return the distance between two points """
    return math.sqrt((pts1[0] - pts2[0])**2 + (pts1[1] - pts2[1])**2)

def angle_between_points(point_1, point_2):
    """ get angle between two points """  
    angle = math.degrees(math.atan2(-(point_2[1] - point_1[1]) , (point_2[0] - point_1[0])))
    return angle

def x_intersection_min(box1, box2):
    """ returns the vertical intersection / minumum width """
    up1 = box1.x1
    down1 = box1.x2
    up2 = box2.x1
    down2 = box2.x2
    messsage = "Error: Not proper box coordinates(xintersection min)"
    assert (box1.x2-box1.x1>=0 and box2.x2-box2.x1>=0 ), messsage
    minh = min( box1.x2-box1.x1 ,box2.x2-box2.x1 )
    maxh = max( box1.x2-box1.x1 ,box2.x2-box2.x1 )
    if box2.x2 < box1.x1 or box2.x2 < box2.x1:
        inter = 0
    else:
        inter = min(down2,down1) - max(up1,up2)
    return [inter/minh, inter/maxh]

def x_intersection_max(box1, box2):
    """ returns the vertical intersection / maximum width"""
    up1 = box1.x1
    down1 = box1.x2
    up2 = box2.x1
    down2 = box2.x2
    message = "Error: Not proper box coordinates(x intersection max )"
    assert (box1.x2-box1.x1>=0 or box2.x2-box2.x1>=0 ), message
    minh = max( box1.x2-box1.x1 ,box2.x2-box2.x1 )
    
    if box2.x2 < box1.x1 or box2.x2 < box2.x1:
        inter = 0
    else:
        inter = min(down2,down1) - max(up1,up2)

    return inter/minh

def y_intersection_min(box1, box2):
    """ returns horizontal intersection / minimum height """
    up1 = box1.y1
    down1 = box1.y2
    up2 = box2.y1
    down2 = box2.y2
    message = "Error: Not proper box coordinates(y intersection min )"
    assert (box1.y2 - box1.y1 >= 0 or box2.y2 - box2.y1 >= 0 ), message
    minh = min( box1.y2 - box1.y1 ,box2.y2 - box2.y1 )
    maxh = max( box1.y2 - box1.y1 ,box2.y2 - box2.y1 )
    if box2.y2 < box1.y1 or box1.y2 < box2.y1:
        inter = 0
    else:
        inter = min(down2,down1) - max(up1,up2)

    return [inter/minh, inter/maxh]

def y_intersection_max(box1, box2):
    """ returns horizontal intersection / maximum height """
    up1 = box1.y1
    down1 = box1.y2
    up2 = box2.y1
    down2 = box2.y2
    message = "Error: Not proper box coordinates(y intersection max )"
    assert (box1.y2 - box1.y1 >= 0 or box2.y2 - box2.y1 >= 0 ), message
    minh = max( box1.y2 - box1.y1 ,box2.y2 - box2.y1 )
    if box2.y2 < box1.y1 or box1.y2 < box2.y1:
        inter = 0
    else:
        inter = min(down2,down1) - max(up1,up2)

    return inter/minh

def distance_centriod(box1, box2):
    """ returns the eucledian distance between the centroids """
    point1 = ((box1.x1 + box1.x2)/2, (box1.y1 + box1.y2)/2 )
    point2 = ((box2.x1 + box2.x2)/2, (box2.y1 + box2.y2)/2 )
    distance = math.sqrt( (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 ) 
    return distance

def angle_between_boxes(box1, box2):
    """ calculate the angle between the source box1 and the target box box2"""
    point_1 = ((box1.x1 + box1.x2)/2, (box1.y1 + box1.y2)/2)
    point_2 = ((box2.x1 + box2.x2)/2, (box2.y1 + box2.y2)/2)
    #angle = (point_2[1] - point_1[1]) , (point_2[0] - point_1[0])
    angle = math.degrees(math.atan2(-(point_2[1] - point_1[1]) , (point_2[0] - point_1[0])))
    return angle

def distance_combination(box1, box2):
    """ returns the minimum distance between points in the give two bounding box, 
    it is expensive """
    b1_points = []
    b1_lines = []
    b1_lines.append([(box1.x1, box1.y1), (box1.x1, box1.y2)])
    b1_lines.append([(box1.x1, box1.y1), (box1.x2, box1.y1)])
    b1_lines.append([(box1.x2, box1.y1), (box1.x2, box1.y2)])
    b1_lines.append([(box1.x1, box1.y2), (box1.x2, box1.y2)])
    b1_points.append((box1.x1, box1.y1))
    b1_points.append((box1.x1, box1.y2))
    b1_points.append((box1.x2, box1.y1))
    b1_points.append((box1.x2, box1.y2))
    for i in b1_lines:
        
        b1_points += get_peripheral_points(i[0], i[1], 5)
    b2_points = []  
    b2_lines = []
    b2_lines.append([(box2.x1, box2.y1), (box2.x1, box2.y2)])
    b2_lines.append([(box2.x1, box2.y1), (box2.x2, box2.y1)])
    b2_lines.append([(box2.x2, box2.y1), (box2.x2, box2.y2)])
    b2_lines.append([(box2.x1, box2.y2), (box2.x2, box2.y2)])
    b2_points.append((box2.x1, box2.y1))
    b2_points.append((box2.x1, box2.y2))
    b2_points.append((box2.x2, box2.y1))
    b2_points.append((box2.x2, box2.y2))
    for i in b2_lines:
        b2_points += get_peripheral_points(i[0], i[1], 5)
    min_dist = 99999
    for i in b1_points:
        for j in b2_points:
            dist = distance_between_points(i, j)
            if min_dist > dist:
                min_dist = dist
    return min_dist + 1e-6

def get_list_of_distances(box1, box2):
    """ get some possible distance value between points in two boxes"""
    b1_points = []
    b1_lines = []
    b1_lines.append([(box1.x1, box1.y1), (box1.x1, box1.y2)])
    b1_lines.append([(box1.x1, box1.y1), (box1.x2, box1.y1)])
    b1_lines.append([(box1.x2, box1.y1), (box1.x2, box1.y2)])
    b1_lines.append([(box1.x1, box1.y2), (box1.x2, box1.y2)])
    b1_points.append((box1.x1, box1.y1))
    b1_points.append((box1.x1, box1.y2))
    b1_points.append((box1.x2, box1.y1))
    b1_points.append((box1.x2, box1.y2))
    for i in b1_lines:
        b1_points += get_peripheral_points(i[0], i[1], 2)
    b2_points = []  
    b2_lines = []
    b2_lines.append([(box2.x1, box2.y1), (box2.x1, box2.y2)])
    b2_lines.append([(box2.x1, box2.y1), (box2.x2, box2.y1)])
    b2_lines.append([(box2.x2, box2.y1), (box2.x2, box2.y2)])
    b2_lines.append([(box2.x1, box2.y2), (box2.x2, box2.y2)])
    b2_points.append((box2.x1, box2.y1))
    b2_points.append((box2.x1, box2.y2))
    b2_points.append((box2.x2, box2.y1))
    b2_points.append((box2.x2, box2.y2))
    for i in b2_lines:
        b2_points += get_peripheral_points(i[0], i[1], 2)
    distance_list = []
    for i in b1_points:
        for j in b2_points:
            dist = distance_between_points(i, j)
            distance_list.append(dist)
    return distance_list


def get_list_of_angles(box1, box2):
    """ get some possible angles between points in two boxes"""
    b1_points = []
    b1_lines = []
    b1_lines.append([(box1.x1, box1.y1), (box1.x1, box1.y2)])
    b1_lines.append([(box1.x1, box1.y1), (box1.x2, box1.y1)])
    b1_lines.append([(box1.x2, box1.y1), (box1.x2, box1.y2)])
    b1_lines.append([(box1.x1, box1.y2), (box1.x2, box1.y2)])
    b1_points.append((box1.x1, box1.y1))
    b1_points.append((box1.x1, box1.y2))
    b1_points.append((box1.x2, box1.y1))
    b1_points.append((box1.x2, box1.y2))
    for i in b1_lines:
        b1_points += get_peripheral_points(i[0], i[1], 2)
    b2_points = []  
    b2_lines = []
    b2_lines.append([(box2.x1, box2.y1), (box2.x1, box2.y2)])
    b2_lines.append([(box2.x1, box2.y1), (box2.x2, box2.y1)])
    b2_lines.append([(box2.x2, box2.y1), (box2.x2, box2.y2)])
    b2_lines.append([(box2.x1, box2.y2), (box2.x2, box2.y2)])
    b2_points.append((box2.x1, box2.y1))
    b2_points.append((box2.x1, box2.y2))
    b2_points.append((box2.x2, box2.y1))
    b2_points.append((box2.x2, box2.y2))
    for i in b2_lines:
        b2_points += get_peripheral_points(i[0], i[1], 2)
    angle_list = []
    for i in b1_points:
        for j in b2_points:
            dist = angle_between_points(i, j)
            angle_list.append(dist)
    return angle_list


class operations:

    def __init__(self, feature_obj : document_properties, image_shape : tuple):

        self.document_features = feature_obj
        self.adjacency_list = {}
        self.links = {}
        self.line_of_sight = {}
        self.image_shape = image_shape

    def get_neighbors(self, num_of_neighbors):
        """ get neighbors entered ... """
        for key_i in self.document_features.document_elements:
            self.adjacency_list[key_i] = []
            bbox_i = self.document_features.document_elements[key_i]
            distance_list = []
            for key_j in self.document_features.document_elements:
                if key_i != key_j:
                    bbox_j = self.document_features.document_elements[key_j]
                    distance_list.append([key_j, distance_combination(bbox_i, bbox_j) ])

           
            distance_list.sort(key=lambda x : x[1])
           
            for k in range(num_of_neighbors):
                if num_of_neighbors < len(distance_list):
                    self.adjacency_list[key_i].append(distance_list[k][0])

        return 1


    def get_line_of_sight(self):
        """ get line of sight json , right neigbor and bottom neigbor """
        h, w, _ = self.image_shape
        for key_i in self.document_features.document_elements:
            self.line_of_sight[key_i] = {}
            self.line_of_sight[key_i]["bottom"] = None
            bb1 = self.document_features.document_elements[key_i]
            temp = []
            for key_j in self.document_features.document_elements:
                if key_i != key_j:
                    bb2 = self.document_features.document_elements[key_j]
                    if y_intersection_min(bb1, bb2) > 0.0:
                        if bb1.x1 < bb2.x1:
                            if  bb2.x1 - bb1.x1 < 0.1 *h:
                                temp.append([key_j, bb2.x1 - bb1.x1])
            temp.sort(key=lambda x:x[1])
            if len(temp) != 0:
                self.line_of_sight[key_i]["bottom"] = temp[0][0]

        for key_i in self.document_features.document_elements:
            self.line_of_sight[key_i]["right"] = None
            bb1 = self.document_features.document_elements[key_i]
            temp = []
            for key_j in self.document_features.document_elements:
                if key_i != key_j:
                    bb2 = self.document_features.document_elements[key_j]
                    if x_intersection_max(bb1, bb2) > 0.5:
                        if bb1.y1 < bb2.y1:
                            if bb2.y1 - bb1.y1 < 0.1 * w:
                                temp.append([key_j, bb2.y1 - bb1.y1])
            temp.sort(key=lambda x:x[1])
            if len(temp) != 0:
                self.line_of_sight[key_i]["right"] = temp[0][0]

        return 1
    
    def get_scoring_features(self, index : int):
        """ add scores to all the neighbors of box """
        scoring_features = {}
        features = self.document_features.document_elements[index]
        neighbors = self.adjacency_list[index]
        for neighbor in neighbors:
            scoring_features[neighbor] = {}
        sum_of_distance = 0
        for neighbor in neighbors:
            sum_of_distance += 1 / distance_combination(features, (self.document_features.document_elements[neighbor]))
        for neighbor in neighbors:
            scoring_features[neighbor]["distance"] = get_list_of_distances(features, self.document_features.document_elements[neighbor])
            scoring_features[neighbor]["distance_score"] = [(1 / distance_combination(features, self.document_features.document_elements[neighbor])) / sum_of_distance]
            scoring_features[neighbor]["angle"] = get_list_of_angles(features, self.document_features.document_elements[neighbor])
            scoring_features[neighbor]["xintersection"] = x_intersection_min(features, self.document_features.document_elements[neighbor])
            scoring_features[neighbor]["yintersection"] = y_intersection_min(features, self.document_features.document_elements[neighbor])
            neighbor_features = self.document_features.document_elements[neighbor]
            scoring_features[neighbor]["bbox"] = [neighbor_features.x1, neighbor_features.y1, neighbor_features.x2, neighbor_features.y2]
            scoring_features[neighbor]["id"] = neighbor
            scoring_features[neighbor]["class"] = neighbor_features.label
        return scoring_features


    def get_scoring(self):
        """" calls the appropriate scoring function """
        scores_dict = {}
        all_index = self.document_features.form_element_id + self.document_features.text_element_id
        for index in all_index:
            scores_dict[index] = {}
            scores_dict[index]["features"] = {}
            scores_dict[index]["neighbors"] = {}
            feautres = self.document_features.document_elements[index]
            scores_dict[index]["features"]["bbox"] = [feautres.x1, feautres.y1, feautres.x2, feautres.y2]
            scores_dict[index]["features"]["class"] = feautres.label
            scores_dict[index]["features"]["id"] = index
            if self.document_features.document_elements[index].label == 0:
                scores_dict[index]["neighbors"] = self.get_scoring_features(index)
            elif self.document_features.document_elements[index].label == 1:
                scores_dict[index]["neighbors"] = self.get_scoring_features(index)
            elif self.document_features.document_elements[index].label == 2:
                scores_dict[index]["neighbors"] = self.get_scoring_features(index)
            elif self.document_features.document_elements[index].label == 3:
                scores_dict[index]["neighbors"] = self.get_scoring_features(index)

        return scores_dict


    def scoring_on_box():
        pass
        

    def add_lines_reading_order(self):
        """ seperate list or dict to give us the lines of each element or reading order """
        pass

    def global_correction(self):
        ## try to resolve conflicts , change it to text ordering
        
        pass

    def get_json_output(self):
        ## return json output links seperately
        
        pass

   