import cv2
import os
import numpy as np


def contours_join(parent_contour, child_contour):
    """
    Join parent contour with child contour
    Donut use case. Inside donut shouldn't detect anything
    """
    def is_clockwise(contour):
        value = 0
        num = len(contour)
        for i in range(len(contour)):
            p1 = contour[i]
            if i < num - 1:
                p2 = contour[i + 1]
            else:
                p2 = contour[0]
            value += (p2[0][0] - p1[0][0]) * (p2[0][1] + p1[0][1])
        return value < 0

    def get_merge_point_idx(contour1, contour2):
        idx1 = 0
        idx2 = 0
        distance_min = -1
        for i, p1 in enumerate(contour1):
            for j, p2 in enumerate(contour2):
                distance = pow(p2[0][0] - p1[0][0], 2) + pow(p2[0][1] - p1[0][1], 2)
                if distance_min < 0:
                    distance_min = distance
                    idx1 = i
                    idx2 = j
                elif distance < distance_min:
                    distance_min = distance
                    idx1 = i
                    idx2 = j
        return idx1, idx2

    def merge_contours(contour1, contour2, idx1, idx2):
        contour = []
        for i in list(range(0, idx1 + 1)):
            contour.append(contour1[i])
        for i in list(range(idx2, len(contour2))):
            contour.append(contour2[i])
        for i in list(range(0, idx2 + 1)):
            contour.append(contour2[i])
        for i in list(range(idx1, len(contour1))):
            contour.append(contour1[i])
        contour = np.array(contour, dtype=np.int32)
        return contour

    def merge_with_parent(parent_contour, contour):
        if not is_clockwise(parent_contour):
            parent_contour = parent_contour[::-1]
        if is_clockwise(contour):
            contour = contour[::-1]
        idx1, idx2 = get_merge_point_idx(parent_contour, contour)
        return merge_contours(parent_contour, contour, idx1, idx2)

    return merge_with_parent(parent_contour=parent_contour, contour=child_contour)


def group_child_contours_with_parent(hierarchy):
    """
    returns:
        {
            parent_key: {
                "parent": parent_key,
                "child": [child_keys]
            }
        }
    """
    groups = {}
    for i, h in enumerate(hierarchy.squeeze()):
        parent_index = h[3]
        if parent_index != -1:
            if groups.get(parent_index) is not None:
                groups[parent_index]["child"].append(i)
            else:
                groups[parent_index] = {"parent": parent_index, "child": [i]}
        else:
            if groups.get(i) is not None:
                groups[i]["parent"] = i
            else:
                groups[i] = {"parent": i, "child": []}
    return groups

def convert_yolo_label_to_mask(original_image_file_path, label_file_path):
    original_image = cv2.imread(original_image_file_path, cv2.IMREAD_GRAYSCALE)
    h, w = original_image.shape
    mask = np.zeros((h, w), np.uint8)

    with open(label_file_path, 'r') as f:
        for line in map(lambda x: x.rsplit(), f.readlines()):
            x_points = list(map(lambda x: int(float(x) * w), line[1::2]))
            y_points = list(map(lambda y: int(float(y) * h), line[2::2]))
            pts = np.array(list(zip(x_points, y_points)),np.int32).reshape((-1, 1, 2))
            # cv2.polylines(mask, [pts], True, 255, 1) # Not filling
            cv2.fillPoly(mask, [pts], 255)
    return mask

def convert_mask_to_yolo_seg_label(mask_path):
    label_str, test_mask = "", None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return label_str, test_mask
    test_mask = np.zeros((height, width), dtype=np.uint8) # Test mask
    if hierarchy.shape[1] > 1: # More than 1 contour
        contour_groups = group_child_contours_with_parent(hierarchy)
        for contour_group in contour_groups.values():
            contour_label = ""
            parent_contour = contours[contour_group["parent"]]
            for child in contour_group["child"]:
                parent_contour = contours_join(parent_contour=parent_contour, child_contour=contours[child])
            contour_to_write = parent_contour.squeeze() # Coords in [[x, y]], [[x, y]] must remove 1 axis
            contour_to_write_list = contour_to_write.tolist()
            if len(contour_to_write_list) < 3: 
                continue # Need at least 3 points to create the contour
            contour_to_write_list = filter(lambda c: isinstance(c, list), contour_to_write_list) # Filter, if contour not a list (only one point)
            for point in contour_to_write_list:
                contour_label += f" {round(float(point[0]) / float(width), 6)}"
                contour_label += f" {round(float(point[1])/ float(height), 6)}"
            if contour_label:
                label_str += f"0 {contour_label}\n"
            parent_contour = np.expand_dims(parent_contour, axis=0)
            cv2.drawContours(test_mask, parent_contour, -1, 255, -1)
    else:
        contour_label = ""
        contour_to_write = (contours[0].squeeze())
        contour_to_write_list = contour_to_write.tolist()
        if len(contour_to_write_list) < 3:
            return label_str, test_mask
        contour_to_write_list = filter(lambda c: isinstance(c, list), contour_to_write_list) # Filter, if contour not a list
        for point in contour_to_write_list:
            contour_label += f" {round(float(point[0]) / float(width), 6)}"
            contour_label += f" {round(float(point[1])/ float(height), 6)}"
        if contour_label:
            label_str += f"0 {contour_label}\n"
        parent_contour = np.expand_dims(contour_to_write, axis=0)
        cv2.drawContours(test_mask, parent_contour, -1, 255, -1)
    label_str = label_str.rstrip()  # Remove last \n
    return label_str, test_mask

def main():
    input_dir = './tmp/val_masks/'
    label_dir = './tmp/val_labels_v3'
    output_dir = './tmp/val_labels_v3'
    for j in os.listdir(input_dir):
        mask_path = os.path.join(input_dir, j)
        label_str, mask_generated = convert_mask_to_yolo_seg_label(mask_path)
        # cv2.imwrite(f"{os.path.join(output_dir, j)}", mask_generated)
        with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f:
            f.write(label_str)
            f.close()
        # with open('mask.txt', "w") as f:
    
    # for j in os.listdir(input_dir):
    #     image_path = os.path.join(input_dir, j)
    #     label_file = j.replace('jpg', 'txt')
    #     label_path = os.path.join(label_dir, label_file)
    #     mask = convert_yolo_label_to_mask(image_path, label_path)
    #     cv2.imwrite(f"{os.path.join(output_dir, j)}", mask)
main()
