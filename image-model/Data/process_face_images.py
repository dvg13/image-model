import cv2
from scipy.misc import imsave
import os
import processing_utils as utils
import sys

def get_largest_face(faces):
    max_face = 0
    max_idx = -1

    for i in range(len(faces)):
        if faces[i][2] > max_face:
            max_face = faces[i][2]
            max_idx = i

    return faces[i]

def crop_images(directory,output_dir,scale):
    not_one_faced = 0

    for dirs,subdirs,files in os.walk(directory):
        for f in files:
            if f.endswith("jpg"):
                image = cv2.imread(os.path.join("lfw",f))
                faces = utils.get_face_bboxes(image)

                if len(faces) != 1:
                    not_one_faced += 1
                else:
                    face = faces[0]

                try:
                    cropped = utils.crop_around_bbox(image,face,scale)
                    imsave(os.path.join(output_dir,f),cropped)
                except:
                    not_one_faced += 1

        return not_one_faced

def main(input_dir,output_dir,scale):
    missed = crop_images(input_dir,output_dir,scale)
    print("No face found for {} images".format(missed))

if __name__ == "__main__":
    input_dir,output_dir = sys.argv[1:3]
    scale = float(sys.argv[3])

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    main(input_dir,output_dir,scale)
