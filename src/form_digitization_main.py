""" testing form field recognition features """
import json
import os
import random
import cv2
from form_digitization import line_grouping
from form_field_recognition import form_field_recognition_main
from text_detection import text_detection_main
from form_digitization import form_digitiztion_main
from form_digitization import bounding_box
from form_digitization import operations


class testing:
    """ test class """
    def __init__(self):
        self.form_field_recognition = form_field_recognition_main.form_field_recognition()
        self.text_detection = text_detection_main.text_detection()

    def process(self, image):
        self.form_field_elements = self.form_field_recognition.recognize(image)
        self.text_field_elements = self.text_detection.detect(image)
        self.list_of_bbox = []
        index = 0
        for bbox in self.form_field_elements:
            if bbox[0] == 0:
                tmp = [bbox[1],bbox[2],bbox[3],bbox[4],int(bbox[0]),"Box"]
                self.list_of_bbox.append(tmp)
            elif bbox[0] == 1:
                tmp = [bbox[1],bbox[2],bbox[3],bbox[4], int(bbox[0]),"CheckBox"]
                self.list_of_bbox.append(tmp)
            elif bbox[0] == 2:
                tmp = [bbox[1],bbox[2],bbox[3],bbox[4], int(bbox[0]),"Line"]
                self.list_of_bbox.append(tmp)

        text_lines, individual_bbox = line_grouping.line_group_heuristics(self.text_field_elements, image.shape)
        for bbox in text_lines:
            self.list_of_bbox.append([bbox[0], bbox[1], bbox[2], bbox[3], int(3), bbox[4]])

        document_object = bounding_box.document_properties(self.list_of_bbox, image.shape)
        #output = document_object.get_document_details_json()
        object_new = operations.operations(document_object, image.shape)
        object_new.get_neighbors(10)
        #object_new.get_line_of_sight()
        output = object_new.get_scoring()
        return output

class  form_digitization:
    """ have loaded models as objects"""
    def __init__(self, ): 
        self.form_field_recognition = form_field_recognition_main.form_field_recognition()
        self.text_detection = text_detection_main.text_detection()

    def process(self, image):
        self.form_field_elements = self.form_field_recognition.recognize(image)
        self.text_field_elements = self.text_detection.detect(image)
        #print(len(self.form_field_elements))
        #print(len(self.text_field_elements))
        ## bounding box initialization
        self.list_of_bbox = []
        index = 0
        for bbox in self.form_field_elements:
            if bbox[0] == 0:
                tmp = bounding_box.bounding_box(bbox[1],bbox[2],bbox[3],bbox[4],bbox[0],"Box", index)
                self.list_of_bbox.append(tmp)
            elif bbox[0] == 1:
                tmp = bounding_box.bounding_box(bbox[1],bbox[2],bbox[3],bbox[4],bbox[0],"CheckBox", index)
                self.list_of_bbox.append(tmp)
            elif bbox[0] == 2:
                tmp = bounding_box.bounding_box(bbox[1],bbox[2],bbox[3],bbox[4],bbox[0],"Line", index)
                self.list_of_bbox.append(tmp)
            index += 1

        #print(self.text_field_elements)
        text_lines, individual_bbox = line_grouping.line_group_heuristics(self.text_field_elements, image.shape)


        #print(len(self.list_of_bbox))
        #print(self.list_of_bbox[0].print_value())

        for bbox in text_lines:
            self.list_of_bbox.append(bounding_box.bounding_box(bbox[0], bbox[1], bbox[2], bbox[3], 3, bbox[4], index))
            index += 1

        #print(len(self.list_of_bbox))
        #print(self.list_of_bbox[-1].print_value())

        self.features = bounding_box.features(self.list_of_bbox, image.shape)
        self.operations = operations.operations(self.features)

        self.operations.get_neighbors(10)

        #print(self.operations)
        return self.operations
        


if __name__ == "__main__":
   
    obj = testing()
  
    files = []
    for r,d,f in os.walk("/home/local/ZOHOCORP/shiva-8700/forms/datasets/pdfimg/pdfimg/"):
        for i in f:
            if i[0] != ".":
                files.append("/home/local/ZOHOCORP/shiva-8700/forms/datasets/pdfimg/pdfimg/" + i)

    #random.shuffle(files)

    already_in = []
    for r,d,f in os.walk("/home/local/ZOHOCORP/shiva-8700/forms/api/updated/src/jsonfiles/"):
        for i in f:
            already_in.append(i)


    for i in files:
        #try:
        
        fn = i.split("/")[-1][:-4] + ".json"
        if fn in already_in:
            continue
        print(fn)
        img = cv2.imread(i)
        output = obj.process(img)
        
        with open( "./jsonfiles/" + fn, "w") as f:
            json.dump(output, f)
        """ 
        except Exception as e:
            with open("error.txt", "a+") as f:
                f.write("==="*10+"\n")
                f.write("file name : " + fn + "\n")
                f.write("\n\n Exception : \n" + str(e) + "\n")
                f.write("==="*10+"\n") """

    # read image
    ## do process
    ## print
    ## plot to check





