import trimesh
import numpy as np
from scipy.misc import imsave
import math
import os
import processing_utils as utils
import sys

def get_depth_map(input_dir,ply_file,output_dir,image_size):

    #load
    mesh = trimesh.load_mesh(os.path.join(input_dir,ply_file))

    #rotate
    rotation_matrix = trimesh.transformations.euler_matrix(0, 0, math.pi/2, 'rxyz')
    mesh.apply_transform(rotation_matrix)

    #get big ass point cloud
    pts = trimesh.sample.sample_surface_even(mesh,100000)

    #needs to be positive for the interpolation function
    pts -= np.min(pts)

    #scale the x and y of the depth maps into the imag
    pts[:,0:2] *= (image_size / np.max(pts[:,0:2] + 1e-5))

    #scale z's to 0,1 - All scaling should be done across the class
    #pts[:,2] /= float(max(pts[:,2]))

    #generate cloud
    image = np.zeros((image_size,image_size))
    for pt in pts:
        image[math.floor(pt[0]),math.floor(pt[1])] = max(pt[2], image[math.floor(pt[0]),math.floor(pt[1])])

    #interpolate_missing(image)
    utils.interpolate_missing_by_row(image)
    imsave(os.path.join(output_dir,ply_file.replace(".ply",".png")),image)

def main(input_dir,output_dir,image_size):
    for dirs,subs,files in os.walk(input_dir):
        for f in files:
            if f.endswith("ply"):
                try:
                    get_depth_map(input_dir,f,output_dir,image_size)
                except:
                    pass

if __name__ == "__main__":
    input_dir,output_dir = sys.argv[1:3]
    image_size = int(sys.argv[3])

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    main(input_dir,output_dir,image_size)
