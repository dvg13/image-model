import numpy as np
import cv2
from scipy.misc import imsave
import scipy
import gzip
import os
import matplotlib.pyplot as plt
import processing_utils as utils
import sys


def save_image(image,fname,output_dir):
    """
    save image with given fname to output directory
    """
    imsave(os.path.join(output_dir,fname),image)

def read_abs(abs_text):
    """
    convert the .abs text into it's component parts
    """
    rows,cols,desc,flags,x,y,z = abs_text.split("\n")

    rows = int(rows.split()[0])
    cols = int(cols.split()[0])

    flags = np.array(flags.split(),dtype=np.int8).astype(np.bool)
    z = np.array(z.split(),dtype=np.float32)

    return rows,cols,flags,z

def get_z(rows,cols,flags,z):
    """
    convert the z string into a depth map
    """
    validz = z[flags==True]
    z -= np.min(validz)
    z[z==np.min(z)] = 0
    #z = z / np.max(z)

    return z.reshape(rows,cols)

def read_depth_map(gz_filename):
    """
    get z channel info from .abs.gz file
    """
    with gzip.open(gz_filename,'rb') as abs_file:
        abs_bytes = abs_file.read()

    abs_string = abs_bytes.decode()
    rows,cols,flags,z = read_abs(abs_string)
    return get_z(rows,cols,flags,z)

def get_next_filename(filename):
    """
    get the ppm file that corresponds to an .abs.gz file
    """
    basename = filename.split(".")[0]
    individual,photo = basename.split("d")
    next_photo = int(photo) + 1
    return individual + "d" + str(next_photo) + ".ppm"


def process_ND_pair(input_dir,output_dir,filename,scale,image_shape=[128,128],gray=True):
    """
    from a filename for a .abs.gz file,
    create proper depth map and crop both images
    about the face.  filename is the name of the .abs.gz file
    """
    depth_image = read_depth_map(os.path.join(input_dir,filename))
    image = cv2.imread(os.path.join(input_dir,get_next_filename(filename)))
    face_bboxes = utils.get_face_bboxes(image)

    if len(face_bboxes):
        bbox = face_bboxes[0]

        bound_image = utils.crop_around_bbox(image,bbox,scale,True)
        bound_depth_image = utils.crop_around_bbox(depth_image,bbox,scale,True)

        utils.interpolate_missing_by_row(bound_depth_image)

        #what are the filenames to print with?  Might change this - but want them to be the same
        basename = filename.split(".")[0]
        image_fname = basename + "_I.png"
        depth_fname = basename + "_D.png"

        save_image(bound_image,image_fname,output_dir)
        save_image(bound_depth_image,depth_fname,output_dir)

def main(input_dir,output_dir,scale):
    for dirs,subs,files in os.walk(input_dir):
        for f in files:
            if f.endswith(".abs.gz"):
                try:
                    process_ND_pair(input_dir,output_dir,f,scale)
                except:
                    pass

if __name__ == "__main__":
    input_dir,output_dir = sys.argv[1:3]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    scale = float(sys.argv[3])

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    main(input_dir,output_dir,scale)
