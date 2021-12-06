""" bounding_box contains both bounding_box and Document_properities classes"""
import cv2
import numpy as np
import math
import copy

class bounding_box:
    """" data structure for a bounding box"""
    def __init__(self, rx1 = -1, ry1 = -1, rx2 = -1, ry2 = -1, rlabel = -1, rvalue = -1 ):   
        assert  isinstance(rx1, int), "Error improper datatypes "
        assert  isinstance(rx2, int), "Error improper datatypes "
        assert  isinstance(ry1, int), "Error improper datatypes "
        assert  isinstance(ry2, int), "Error improper datatypes "
        assert  isinstance(rlabel, int), "Error improper datatypes "
        assert  isinstance(rvalue, str), "Error improper datatypes "
        assert rx1 < rx2, "Error: improper coordinates x"
        assert ry1 < ry2, "Error: improper coordinates y"
        assert rlabel  in [0, 1, 2, 3], "Error: improper label"
        self.x1 = rx1
        self.y1 = ry1
        self.x2 = rx2
        self.y2 = ry2
        self.label = rlabel
        self.value = rvalue
        
    def print_value(self,):
        """ display function to display individual bounding box """
        if self.x1 == -1:
            print("No data has be added to this obj")
        else:
            print("Coordinates : (",self.x1, self.y1,"), (", self.x2, self.y2, ")" )
            print("Class : ", self.label)
            print("Value : ", self.value.encode("utf-8"))

class document_properties:
    """ document representation """
    def __init__(self, list_of_bbox, image_shape):
        self.document_elements = {}
        self.form_element_id = []
        self.text_element_id = []

        for box in list_of_bbox:
            self.document_elements[list_of_bbox.index(box)] = bounding_box(box[0], box[1], box[2], box[3], box[4], box[5])
            if box[4] == 3:
                self.text_element_id.append(list_of_bbox.index(box))
            else:
                self.form_element_id.append(list_of_bbox.index(box))
        self.image_shape = image_shape
        self.average_height, self.average_width = self.get_averages()
        self.label_dict = self.get_label_count()

    def get_averages(self):
        height, width = 0, 0
        if len(self.document_elements) == 0:
            return height, width
        for key in self.document_elements:
            height += self.document_elements[key].y2 - self.document_elements[key].y1
            width += self.document_elements[key].x2 - self.document_elements[key].x1
        height /= len(self.document_elements)
        width /= len(self.document_elements)
        return height, width

    def get_label_count(self):
        temp = {}
        for key in self.document_elements:
            if self.document_elements[key].label not in temp:
                temp[self.document_elements[key].label] = []
            temp[self.document_elements[key].label].append(key)
        return temp

    def print_value(self):
        print("Document has ===== ")
        print("- number of form elements : ", len(self.form_element_id))
        print("- number of text elements : ", len(self.text_element_id))
        print("- average height, width : ", self.average_height, self.average_width)
        print("- label counts : ")
        for key in self.label_dict:
            print("  -- label ", key, " has ", len(self.label_dict[key]), " elements")

    def get_individual_elements(self, key):
        tmp = {}
        bbox = self.document_elements[key]
        tmp["bounding_box"] = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
        tmp["label"] = bbox.label
        tmp["value"] = bbox.value
        return tmp

    def get_document_details_json(self):
        document_details = {}
        document_details["elements"] = {}
        for key in self.document_elements:
            document_details["elements"][key] = self.get_individual_elements(key)
        document_details["form_id"] = self.form_element_id
        document_details["text_id"] = self.text_element_id
        document_details["text_count"] = len(self.text_element_id)
        if 0 in self.label_dict:
            document_details["box_count"] = len(self.label_dict[0])
        if 1 in self.label_dict:
            document_details["checkbox_count"] = len(self.label_dict[1])
        if 2 in self.label_dict:
            document_details["line_count"] = len(self.label_dict[2])
        return document_details

   