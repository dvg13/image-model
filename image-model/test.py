import numpy as np
import argparse
import os.path
import sys
from scipy.misc import imsave

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import model_test
import single_reader as reader

from skimage.color import gray2rgb

# Basic model parameters as external flags
FLAGS = None

def get_placeholder(batch_size,channels):
    """
    returns a placeholder given the dimensions.  Only works with square images
    """
    images_placeholder = tf.placeholder(tf.float32, shape=[batch_size,FLAGS.image_size,
                                                    FLAGS.image_size,FLAGS.channels])
    return images_placeholder

def fix_image(image):
    image += np.min(image)
    return images if image.shape[-1] == 3 else np.concatenate(gray2rgb(image),axis=1)

def print_images(images,output_dir,image_num=0,pair=False,synth_images=None):
    """
    outputs n generated images at a given step
    """
    for i in xrange(images.shape[0]):
        to_print = fix_image(images[i])

        if pair and synth_images is not None:
            synth_to_print = fix_image(synth_images[i])
            to_print = np.hstack((to_print,synth_to_print))

        #What is the name of the image?
        imsave(os.path.join(output_dir,str(image_num + i) + ".png"), to_print)

def get_test_fetch(synth_placeholder,model):
    """
    get the generated images variable from the tensorflow graph
    """
    with tf.variable_scope("G"):
        generated_images = model.generator(synth_placeholder,FLAGS.channels,is_training=False)
    return generated_images

def test():
    """
    Just run images through generator.  Requires for the proper generator arch to be Flagged
    """
    image_reader = reader.ImageReader(FLAGS.test_dir,FLAGS.image_size,FLAGS.channels)
    num_images = image_reader.num_images()
    model = model_test.GanModel(FLAGS.batch_size,FLAGS.image_size,FLAGS.gen_arch,FLAGS.batch_norm,training=False)

    graph = tf.Graph()
    with graph.as_default():
        placeholder = get_placeholder(FLAGS.batch_size,FLAGS.channels)
        test_fetch = get_test_fetch(placeholder, model)

        with tf.Session() as session:
            if FLAGS.load is not None:
                saver=tf.train.Saver()
                saver.restore(session, FLAGS.load)
            else:
                print("Need to specify a valid model to load:  --load=path")
                return

            #need to loop based on the size of the test set
            for i in range(0,num_images,FLAGS.batch_size):

                synth_batch = image_reader.next(min(FLAGS.batch_size,num_images-1))
                feed_dict = {placeholder:synth_batch}
                generated_images = session.run(test_fetch,feed_dict=feed_dict)

                #write generated_images to file
                print_images(generated_images,FLAGS.output_dir,i,FLAGS.pair_images,synth_batch)

def main(_):
    #create output directory - Do we want to give these names?  Based on the checkpoint
    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_dir',
        type=str,
        default='../Data/Faces_depth_test',
        help='Directory of Synthetic Images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../Data/L1_test',
        help='Directory of Synthetic Images'
    )
    parser.add_argument(
        '--load',
        type=str,
        default=None,
        help='Checkpoint file to load'
    )
    parser.add_argument(
        '--gen_arch',
        type=str,
        default='U',
        help='Shape of Generator achitecture'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=128,
        help='Image Size'
    )
    parser.add_argument(
        '--channels',
        type=int,
        default=1,
        help='Number of channels in the input image'
        )

    parser.add_argument(
        '--batch_norm',
        action='store_true',
        default=False,
        help='Use batch norm for all layers'
    )
    parser.add_argument(
        '--pair_images',
        action='store_true',
        default=False,
        help='Use batch norm for all layers'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
