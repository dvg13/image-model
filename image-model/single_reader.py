import tensorflow as tf
import numpy as np
from queue import Queue
import os
from scipy.misc import imread
from scipy.ndimage.interpolation import zoom
import random
from skimage.color import rgb2gray,gray2rgb

class ImageReader():
    def __init__(self,directory,image_size,channels=1):
        self.directory = directory
        self.image_size = image_size
        self.channels = channels

        self.filenames = self.get_filenames(self.directory)
        self.queue = Queue()
        self.fill_queue(self.queue,self.filenames)

    def get_filenames(self,directory):
        for dirs,subs,files in os.walk(directory):
            return [os.path.join(directory,f) for f in files if (f.endswith("png") or f.endswith("jpg"))]

    def fill_queue(self,q,filenames):
        np.random.shuffle(filenames)
        for f in filenames:
            q.put(f)

    def get_batch(self,q,filenames,channels,batch_size):
        #image_batch = np.zeros((self.batch_size,self.image_size,self.image_size,3))
        image_batch = np.zeros((batch_size,self.image_size,self.image_size,1),dtype=np.float32)

        for i in range(batch_size):

            if q.empty():
                self.fill_queue(q,filenames)

            image = imread(q.get(),mode='F')

            #resize
            image = zoom(image, (float(self.image_size) / image.shape[0],
                                float(self.image_size) / image.shape[1]))

            if channels == 1:
                #now we want them gray
                if len(image.shape) == 2:
                    image.resize(self.image_size,self.image_size,1)
                elif len(image.shape) == 3:
                    image = rgb2gray(image).reshape(self.image_size,self.image_size,1)

            #rescale the values
            image = image.astype(np.float32)
            image -= np.min(image)
            image /= np.max(image)
            image -= np.mean(image)
            image_batch[i] = image

        return image_batch

    def next(self,batch_size):
        images = self.get_batch(self.queue,self.filenames,self.channels,batch_size)
        return images

    def num_images(self):
        return len(self.filenames)

if __name__ == "__main__":
    pass
    #reader = ImageReader('../Data/Faces_depth','../Data/lfw', 4, 28)
    #result = reader.next(True)
    #for i in range(1000):
    #    result = reader.next(True)
    #    result = reader.next(False)
