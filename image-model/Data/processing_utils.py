import numpy as np
import cv2

def get_face_bboxes(image):
    """
    get the face bounding box from the rgb image.  Scale
    sets the factor by which to increase the size of the box
    in order to better align with the face
    """
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    return faces

def crop_around_bbox(image,bbox,scale,rows_first=False):
    (x, y, w, h) = bbox
    center = (int(x + w/2),int(y+h/2))

    if rows_first:
        face_image = image[center[1]-int(h*scale/2):center[1]+int(h*scale/2),
                          center[0]-int(w*scale/2):center[0]+int(w*scale/2)]
    else:
        face_image = image[center[0]-int(w*scale/2):center[0]+int(w*scale/2),
                          center[1]-int(h*scale/2):center[1]+int(h*scale/2)]

    return face_image

def interpolate_missing_by_row(image):
    """
    if a point has non-zero values to it's left and right, fill it in with the
    average
    """
    for row in range(image.shape[0]):
        nonzero = np.nonzero(image[row])[0]
        if len(nonzero) and len(nonzero) < nonzero[-1] - nonzero[0] + 1:
            for i in range(len(nonzero)-1):
                length = nonzero[i+1] - nonzero[i]
                if  length > 1:
                    left_val = image[row,nonzero[i]]
                    right_val = image[row,nonzero[i+1]]
                    for col in range(nonzero[i]+1,nonzero[i+1]):
                        right_ratio = (col - nonzero[i]) / float(length)
                        image[row,col] = (1-right_ratio) * left_val + right_ratio * right_val

if __name__ == "__main__":
    pass





    #old interpolation code
    # for i in range(1,image.shape[0]-1):
    #     first = None
    #     last = None
    #
    #     for j in range(1,image.shape[1]-1):
    #         if image[i,j] > 0:
    #             if first is None:
    #                 first = j
    #             if j < image.shape[1] - 1 and image[i,j+1] == 0:
    #                 last = j
    #             elif j == image.shape[1] - 1:
    #                 last = j
    #
    #     if first is not None and last is not None:
    #         for j in range(first,last):
    #             if image[i,j] == 0:
    #                 total=0
    #                 n=0
    #                 if i > 0 and image[i-1,j] > 0:
    #                     total += image[i-1,j]
    #                     n+=1
    #                 if i < image.shape[0] - 1 and image[i+1,j] > 0:
    #                     total += image[i+1,j]
    #                     n+=1
    #                 if j > 0 and image[i,j-1] > 0:
    #                     total += image[i,j-1]
    #                     n+=1
    #                 if j < image.shape[1] -1 and image[i,j+1] > 0:
    #                     total += image[i,j+1]
    #                     n+=1
    #
    #                 image[i,j] = float(total) / n
