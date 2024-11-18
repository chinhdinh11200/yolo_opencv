import os
import cv2

base_folder = '/home/chinh/Documents/test-image-data/CelebAMask-HQ/CelebAMask-HQ-mask-anno'

def create_mask(name_folder):
    input_dir = f"{base_folder}/{name_folder}/"

    segment = 2000;
    start = name_folder * segment
    end = (name_folder + 1) * segment

    for i in range(start, end):
        files_need_combine = []
        for file in os.listdir(input_dir):
            if str(i).rjust(5, '0') in file and checkNotIncludes(file):
                files_need_combine.append(file)
        combineFiles(i, input_dir, files_need_combine)

def combineFiles(i, input_dir, files):
    output_dir = './tmp/masks'
    if len(files) > 0:
        combine_mask = cv2.imread(input_dir + files[0])
        for file in files[1:]:
            mask_file = cv2.imread(input_dir + file)
            combine_mask = cv2.bitwise_or(combine_mask, mask_file)
        original_height, original_width = combine_mask.shape[:2]
        new_width = original_width * 2
        new_height = original_height * 2
        resized_combine_mask = cv2.resize(combine_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"{output_dir}/{i}.png", resized_combine_mask)

def checkNotIncludes(file):
    part_ignored = ['neck', 'brow', 'eye', 'lip', 'mouth', 'nose', 'cloth']
    for ignore in part_ignored:
        if ignore in file:
            return False
    return True;

def main():
    num_folders = 5
    for num_folder in range(0, num_folders):
        create_mask(num_folder)

main()
