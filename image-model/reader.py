import tensorflow as tf
import numpy as np
from queue import Queue
import os
from scipy.misc import imread
from scipy.ndimage.interpolation import zoom
import random
from skimage.color import rgb2gray,gray2rgb

class ImageReader():
    def __init__(self,synth_directory,real_directory,image_size,synth_channels=1,real_channels=1):
        self.synth_directory = synth_directory
        self.real_directory = real_directory
        self.image_size = image_size
        self.synth_channels = synth_channels
        self.real_channels = real_channels


        self.synth_filenames = self.get_filenames(self.synth_directory)
        self.real_filenames = self.get_filenames(self.real_directory)

        self.synth_queue = Queue()
        self.real_queue = Queue()

        self.fill_queue(self.synth_queue,self.synth_filenames)
        self.fill_queue(self.real_queue,self.real_filenames)

    def get_filenames(self,directory):
        for dirs,subs,files in os.walk(directory):
            return [os.path.join(directory,f) for f in files if (f.endswith("png") or f.endswith("jpg"))]

    def fill_queue(self,q,filenames):
        np.random.shuffle(filenames)
        for f in filenames:
            q.put(f)

    def get_batch(self,q,filenames,channels,batch_size):
        image_batch = np.zeros((batch_size,self.image_size,self.image_size,channels))

        for i in range(batch_size):

            if q.empty():
                self.fill_queue(q,filenames)

            image = imread(q.get(),mode='F')

            #resize
            image = zoom(image, (float(self.image_size) / image.shape[0],
                                 float(self.image_size) / image.shape[1]))

            #when we wanted them in color
            if channels==3:
                if len(image.shape) == 2:
                    image = gray2rgb(image)

            elif channels==1:
                if len(image.shape) == 2:
                    image.resize(self.image_size,self.image_size,1)
                elif len(image.shape) == 3:
                    image = rgb2gray(image).reshape(self.image_size,self.image_size,1)

            #rescale the values
            image = image.astype(np.float32)
            image -= np.min(image)

            if np.max(image) <= 0:
                i -= 1

            else:
                image /= np.max(image)
                image -= np.mean(image)
                image_batch[i] = image

        return image_batch

    def next(self,batch_size,sample_synth):
        if sample_synth:
            synth_images = self.get_batch(self.synth_queue,self.synth_filenames,
                                          self.synth_channels,batch_size)
            return synth_images
        else:
            real_images = self.get_batch(self.real_queue,self.real_filenames,
                                         self.real_channels,batch_size)
            return real_images


if __name__ == "__main__":
    pass
