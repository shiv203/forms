import os
import cv2
import numpy as np
import json
import time

def present_inside_box(outerbox : list, innerbox : list) -> bool:
    """ return true if outer box encloses the inner box"""
    if outerbox[0] <= innerbox[0] and \
        outerbox[1] <= innerbox[1] and \
            outerbox[2] >= innerbox[2] and outerbox[3] >= innerbox[3]:
            return True
    else:
        return False

def combine_boxes(box_1 : list, box_2 : list ) -> list:
    """ finds the union of two boxes and concatenates the text """
    if box_1[0] < box_2[0]:
        x1 = min(box_1[0], box_2[0])
        y1 = min(box_1[1], box_2[1])
        x2 = max(box_1[2], box_2[2])
        y2 = max(box_1[3], box_2[3])
        s = box_1[4] + " "  + box_2[4]
        return [x1, y1, x2, y2, s]
    else:
        x1 = min(box_1[0], box_2[0])
        y1 = min(box_1[1], box_2[1])
        x2 = max(box_1[2], box_2[2])
        y2 = max(box_1[3], box_2[3])
        s = box_2[4] + " "  + box_1[4]
        return [x1, y1, x2, y2, s]


def line_score(box_1, box_2):
    """ intersection score to decide whether boxes belongs to the same line"""
    up1, down1 = box_1[1], box_1[3]
    up2, down2 = box_2[1], box_2[3]
    min_height = min(down2 - up2, down1 - up1)
    if down2 < up1 or down1 < up2:
        inter = 0
    else:
        inter = min(down2, down1) - max(up1, up2)
    return inter / min_height

def horizontal_association_check(box_1, box_2, average_width, image_shape):
    """ calculate the horizontal bool to concatenate words to text blocks"""
    width_1 = box_1[2] - box_1[0]
    height_1 = box_1[3] - box_1[1]
    width_2 = box_2[2] - box_2[0]
    height_2 = box_2[3] - box_2[1]
    if box_1[0] - box_2[0] > 0:
        length_inbetween = box_1[0] - box_2[2]
    else:
        length_inbetween = box_2[0] - box_1[2]

    if length_inbetween < average_width / 2 and height_1 < 3 * width_1:
        if height_2 < 3 * width_2 and line_score(box_1, box_2) > 0.7:
            return True
    else:
        return False


def line_group_heuristics(list_of_bbox : list, image_shape : tuple):
    """ returns the grouped lines and input boxes """
    avg_height, avg_width = 0, 0
    #print("length - ", len(list_of_bbox))
    for i in range(len(list_of_bbox)):
        avg_height += list_of_bbox[i][3] - list_of_bbox[i][1]
        avg_width += list_of_bbox[i][2] - list_of_bbox[i][0]
    avg_height /= len(list_of_bbox)
    avg_width /= len(list_of_bbox)
    list_of_bbox = sorted(list_of_bbox, key=lambda x:x[1])
    line_groups = []
    current_line = []
    current_bbox = list_of_bbox[0]
    current_line.append(current_bbox)
    for i in range(1, len(list_of_bbox)):
        if line_score(current_bbox, list_of_bbox[i]) > 0.5:
            current_bbox = combine_boxes(list_of_bbox[i], current_bbox)
            current_line.append(list_of_bbox[i])
        else:
            line_groups.append(current_line)
            current_line = [list_of_bbox[i]]
            current_bbox = list_of_bbox[i]
    
    #print("number of lines ", len(line_groups))


    box_list = []
    for i in line_groups:
        if len(i) == 0:
            exit()
        elif len(i) == 1:
            box_list.append(i[0])
        elif len(i) == 2:
            #print("length 2 - ", i)
            if horizontal_association_check(i[0], i[1], avg_width, image_shape):
                box_list.append(combine_boxes(i[0], i[1]))
            else:
                box_list.append(i[0])
                box_list.append(i[1])
        else:
            #print("lines count - ", len(i))
            #print(i[0],"|" ,len(i[1]))
            temp_list = sorted(i, key= lambda x:x[0])
            while len(temp_list) > 0:
                current_bbox = temp_list[0]
                temp_list.remove(current_bbox)
                add_new_ele = []
                for k in range(len(temp_list)):
                    if horizontal_association_check(temp_list[k], current_bbox, avg_width, image_shape):
                        new_box = combine_boxes(temp_list[k], current_bbox)
                        current_bbox = new_box
                        add_new_ele.append(temp_list[k])
                box_list.append(current_bbox)
                for k in add_new_ele:
                    temp_list.remove(k)

    return box_list, list_of_bbox


def combine_bbox(b1, b2):
    x1 = min(b1[0], b2[0])
    y1 = min(b1[1], b2[1])
    x2 = max(b1[2], b2[2])
    y2 = max(b1[3], b2[3])
    s = b1[4] + " " + b2[4]
    return [x1, y1, x2, y2, s]

def same_line_check(b1, b2):
    if b1[1] + (b1[3] - b1[1])*0.01 <= b2[3] - (b2[3] - b2[1])*0.01  and b1[3] - (b1[3] - b1[1])*0.01 >= b2[1] + (b2[3] - b2[1])*0.01:
        return True
    return False


def line_grouping(bbox):
    tmp = []
    for i in bbox:
        tmp.append(i)
        
    tmp.sort(key=lambda x : x[1])
    
    curr_box = tmp[0]
    tmp.remove(curr_box)
    curr_count = 1
   
    lines = []
    while curr_count < len(bbox) and len(tmp) > 0:
        new_line = []
        new_line.append(curr_box)
        remove_list = []
        for i in tmp:
            if same_line_check(i, curr_box):
                curr_count += 1
                new_line.append(i)
                remove_list.append(i)
                
        for i in remove_list:
            tmp.remove(i)
            
        lines.append(new_line)
        if len(tmp) != 0:
            curr_box = tmp[0]
            tmp.remove(curr_box)

    all_lines = []
    c = 0

    #for line in lines:
     #   if len(line) == 35:
      #      for i in line:
       #         print(len(i), " --> ", i)

    
    for line in lines:
        
        #print(len(line))
        if len(line) == 0:
            continue
        elif len(line) == 1:
            all_lines.append(line[0])
        else:
            curr_line = line[0]
            for i in range(1, len(line)):
                curr_line = combine_bbox(curr_line, line[i])
            all_lines.append(curr_line)
        
        

    """    
    for line in lines:
        
        print(len(line))
        if len(line) > 0:
            curr_line = line[0]
            for val in line[1:]:
              c += 1
              print(val)
              curr_line = combine_bbox(curr_line, val)
              c += 1
              if c == 1000:
                raise ValueError(curr_line)
            all_lines.append(curr_line)
        

        #print(len(line), len(all_lines), all_lines[-1][-1]) """


    return all_lines, lines

