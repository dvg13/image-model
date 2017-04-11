import numpy as np
import argparse
import os.path
import sys
import time
from scipy.misc import imsave

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import model_test
import reader
import nd_reader
import replay_cache
import re
from skimage.color import gray2rgb

# Basic model parameters as external flags.  Not sure where this goes with flags.
FLAGS = None

class Train():
    def __init__(self):
        self.RUNID = str(int(time.time()))
        self.d_loss = 0
        self.g_loss = 0
        self.l1_loss = 0
        self.r_loss = 0
        self.difference = 0
        self.run_d = False

    def get_placeholder(self,batch_size):
        """
        returns a placeholder given the dimensions.  Only works with square images
        """

        #print([FLAGS.batch_size,FLAGS.image_size,FLAGS.image_size,FLAGS.channels])
        images_placeholder = tf.placeholder(tf.float32, shape=[batch_size,FLAGS.image_size,
                                                        FLAGS.image_size,FLAGS.channels])
        return images_placeholder

    def print_images(self,images,synth_images,step):
        """
        outputs n generated images at a given step
        """
        if not os.path.exists(os.path.join(FLAGS.log_dir,self.RUNID,str(step))):
            os.mkdir(os.path.join(FLAGS.log_dir,self.RUNID,step))

        for i in range(min(FLAGS.gen_num,FLAGS.batch_size)):
            images[i] += np.min(images[i])
            synth_images[i] += np.min(synth_images[i])
            self.print_image(images[i],synth_images[i],
                os.path.join(FLAGS.log_dir,self.RUNID,step,str(i) + ".png"))

    def print_image(self,image,synth_image,name):
        """
        Prints an image that is a concatenation of the synthetic image and the generated iamge
        """
        im_to_print = image if image.shape[-1] == 3 else np.concatenate(gray2rgb(image),axis=1)
        synth_to_print = synth_image if synth_image.shape[-1] == 3 else np.concatenate(gray2rgb(synth_image),axis=1)
        to_print = np.hstack((im_to_print,synth_to_print))
        imsave(name, to_print)


    #this should be a flag to choose optimizer (prolly just ADAM and RMS Prop)
    def get_train_op(self,loss,learning_rate,update_vars,mode):
        """
        adds a training operation to the tensorflow graph.
        might need to do the update_vars in order to work with batch norm
        """
        if FLAGS.optim == "ADAM":
            optimizer = tf.train.AdamOptimizer(learning_rate)

        elif FLAGS.optim == "RMSP":
            optimizer = tf.train.RMSPropOptimizer(learning_rate)

        global_step = tf.Variable(0,name="global_step")

        if FLAGS.wgan:
            loss = -loss

        train_op = optimizer.minimize(loss, global_step, var_list = update_vars,name=mode)

        return train_op

    def get_l1_fetches(self,synth_placeholder,real_placeholder,model):
        """
        get the variables to apply L1 loss on the generator
        """
        l1_loss,generated_images = model.l1_g(synth_placeholder,real_placeholder,FLAGS.channels)
        g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="G")

        l1_loss = l1_loss * FLAGS.L1_weight

        with tf.variable_scope("L1"):
            l1_train_op = self.get_train_op(l1_loss,FLAGS.llr,g_vars,"L1")
            return l1_train_op,l1_loss,generated_images

    def get_recon_fetches(self,synth_placeholder,model):
        r_loss,recon_images = model.recon_g(synth_placeholder,FLAGS.channels)
        g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="G")
        r_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="R")

        with tf.variable_scope("R"):
            r_train_op = self.get_train_op(r_loss,FLAGS.rlr,g_vars + r_vars,"R")
            return r_train_op,r_loss,recon_images

    def get_d_fetches(self,model,synth_placeholder,real_placeholder,replay_placeholder):
        """
        return the necessary graph variables.  Note that wgan returns an extra
        operation to clip the weights
        the variables returned are:
        [d_loss,g_loss,generated_images,real_preds,synth_preds,difference]
        """
        graph_vars = model.gan(synth_placeholder,real_placeholder,FLAGS.channels,
                               FLAGS.L2gan,FLAGS.wgan,replay_placeholder)
        d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="D")
        d_train_op = self.get_train_op(graph_vars[0],FLAGS.dlr,d_vars,"D")

        return [d_train_op,graph_vars[0]] + graph_vars[3:]

    def get_g_fetches(self,synth_placeholder,real_placeholder,model):
        """
        return the necessary graph variables.
        the variables returned are:
        [d_loss,g_loss,generated_images,real_preds,synth_preds,difference]
        """
        graph_vars = model.gan(synth_placeholder,real_placeholder,FLAGS.channels,FLAGS.L2gan,FLAGS.wgan)
        g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="G")

        g_train_op = self.get_train_op(graph_vars[1],FLAGS.dlr,g_vars,"G")
        return [g_train_op] + graph_vars[1:3]

    def add_replay(self,feed_dict):
        """
        add the replay placeholder.  If it is empty have to add something,
        doesn't really matter what - so we'll add the synth_batch
        """
        if self.replay.nelements > FLAGS.batch_size/2:
            replay_batch = self.replay.next()
            feed_dict[self.replay_placeholder] = replay_batch
        else:
            synth_batch = feed_dict[self.synth_placeholder]
            feed_dict[self.replay_placeholder] = synth_batch[int(FLAGS.batch_size/2):]

    #think about how to set these.
    def get_feed_dict(self,use_replay=False):
        synth_batch = self.image_reader.next(FLAGS.batch_size,True)
        real_batch = self.image_reader.next(FLAGS.batch_size,False)
        feed_dict = {self.synth_placeholder:synth_batch, self.real_placeholder:real_batch}

        if use_replay:
            self.add_replay(feed_dict)

        return feed_dict

    def get_nd_feed_dict(self):
        synth_batch,real_batch=self.nd_image_reader.next(FLAGS.batch_size)
        return{self.synth_placeholder:synth_batch, self.real_placeholder:real_batch}

    def l1(self,l1_fetches,feed_dict):
            _,self.l1_loss,self.generated, = self.session.run(l1_fetches,feed_dict=feed_dict)
            #self.d_loss ,self.difference = self.d_pass(d_fetches,feed_dict)

    def recon(self,recon_fetches,feed_dict):
        _,self.r_loss,self.reconstructed = self.session.run(recon_fetches,feed_dict=feed_dict)


    def wgan(self,d_fetches,g_fetches,g_feed_dict,step):
        """
        run the model as outlined in the wgan paper.  We are lowering the number
        of steps for which to do the high number of D steps, and reduceing 100
        D passes to 50
        """
        d_passes = FLAGS.diter
        if step < 500 or step % 1000 == 0:
            d_passes = 50

        for i in xrange(d_passes):
            if i == 0:
                self.d_loss,self.difference = self.d_pass(d_fetches,g_feed_dict)
            else:
                self.d_loss,self.difference = self.d_pass(d_fetches,self.get_feed_dict(FLAGS.use_replay))
        self.g_loss,self.generated = self.g_pass(g_fetches,g_feed_dict)

    def basic_gan(self,d_fetches,g_fetches,g_feed_dict):
        """
        Will generally run one d pass and one g pass, unless
        the difference in positive and negative predictions is very
        large or very small.  Will loop until 1 g pass has been performed
        """
        ran_d = 0

        #run some number of d passes
        while (np.mean(self.difference) < FLAGS.diff_high) \
                and (ran_d < 1 or np.mean(self.difference) < FLAGS.diff_low):
            if ran_d == 0:
                self.d_loss,self.difference = self.d_pass(d_fetches,g_feed_dict)
            else:
                if FLAGS.use_nd_gan:
                    self.d_loss,self.difference = self.d_pass(d_fetches,self.get_nd_feed_dict())
                else:
                    self.d_loss,self.difference = self.d_pass(d_fetches,self.get_feed_dict(FLAGS.use_replay))
            ran_d += 1

        #run 1 g pass
        self.g_loss,self.generated = self.g_pass(g_fetches,g_feed_dict)

        #run a d pass without updating so that the loss and difference change
        if ran_d == 0:
            self.d_loss,self.difference = self.d_pass(d_fetches,g_feed_dict,False)

    #will need to adopt this for batch norm
    def d_pass(self,d_fetches,feed_dict,training=True):
        if training:
            _,d_loss,*rest,difference = self.session.run(d_fetches,feed_dict=feed_dict)
        else:
            d_loss,*rest,difference = self.session.run(d_fetches[1:],feed_dict=feed_dict)
        return d_loss,difference

    def g_pass(self,g_fetches,feed_dict):
        _,g_loss,generated,*rest = self.session.run(g_fetches,feed_dict=feed_dict)
        return g_loss,generated

    def print_generated(self,synth_batch,step):
        if self.generated is not None:
            self.print_images(self.generated,synth_batch,"step_" + str(step))

    def checkpoint(self,step):
        checkpoint_file = os.path.join(FLAGS.log_dir, self.RUNID,'model.ckpt')
        self.saver.save(self.session, checkpoint_file, global_step=step)

    def load_from_checkpoint(self):
        step = int(FLAGS.load.split("-")[-1])
        self.saver.restore(self.session, FLAGS.load)

        return step

    #change print_freq to display_freq
    def display(self,step):
        output = "step: {} ".format(step)

        if FLAGS.gan:
            output += " d_loss: {} g_loss: {} diff: {}".format(self.d_loss,self.g_loss,np.mean(self.difference))
        if FLAGS.L1:
            output += " L1: {}".format(self.l1_loss)
        if FLAGS.use_recon:
            output += " R: {}".format(self.r_loss)

        print(output)

    def run_summary(self,fetch,feed_dict,step):
        summary = self.session.run(fetch,feed_dict=feed_dict)
        self.summary_writer.add_summary(summary,global_step=step)

    def get_summary_fetch(self,scope):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,scope=scope)
        if len(summaries):
            return tf.summary.merge(summaries)

    def add_summaries(self,step,super_feed_dict,unsuper_feed_dict):
        """
        add the tensorflow summaries associated with the various losses
        The passed in feed dicts can exist or can be none
        """
        l1_sum = self.get_summary_fetch("L1")
        r_sum = self.get_summary_fetch("R")
        gan_sum = self.get_summary_fetch("G|D")

        if l1_sum is not None:
            feed_dict = super_feed_dict if FLAGS.use_nd_L1 else unsuper_feed_dict
            self.run_summary(l1_sum,feed_dict,step)

        if r_sum is not None:
            self.run_summary(r_sum,unsuper_feed_dict,step)

        if gan_sum is not None:
            feed_dict = super_feed_dict if FLAGS.use_nd_gan else unsuper_feed_dict
            self.run_summary(gan_sum,feed_dict,step)

    def run_training(self):
        """
        run training.  Most of the variables are set in the flags.  As of now
        we have two potential training policies that have some repetitive code
        """
        model = model_test.GanModel(FLAGS.batch_size,FLAGS.image_size,FLAGS.gen_arch,FLAGS.batch_norm)

        if FLAGS.gan and not FLAGS.use_nd_gan or FLAGS.L1 and not FLAGS.use_nd_L1:
            self.image_reader = reader.ImageReader(FLAGS.synth_dir,FLAGS.real_dir,FLAGS.image_size)

        if FLAGS.use_nd_L1 or FLAGS.use_nd_gan:
            self.nd_image_reader = nd_reader.NDReader(FLAGS.nd_dir,FLAGS.image_size)

        if FLAGS.use_replay:
            self.replay = replay_cache.ReplayCache(FLAGS.cache_size,FLAGS.batch_size/2,
                                    [FLAGS.image_size,FLAGS.image_size,FLAGS.channels],
                                    FLAGS.cache_path, FLAGS.reuse_replay)

        graph = tf.Graph()
        with graph.as_default():
            self.synth_placeholder = self.get_placeholder(FLAGS.batch_size)
            self.real_placeholder = self.get_placeholder(FLAGS.batch_size)
            self.replay_placeholder = self.get_placeholder(FLAGS.batch_size/2) if FLAGS.use_replay else None

            if FLAGS.L1:
                l1_fetches = self.get_l1_fetches(self.synth_placeholder,self.real_placeholder,model)

            if FLAGS.use_recon:
                recon_fetches = self.get_recon_fetches(self.synth_placeholder,model)

            if FLAGS.gan:
                d_fetches = self.get_d_fetches(model,self.synth_placeholder,self.real_placeholder, self.replay_placeholder)
                g_fetches = self.get_g_fetches(self.synth_placeholder,self.real_placeholder,model)

            init = tf.global_variables_initializer()
            summary_fetch = tf.summary.merge_all()

            self.saver = tf.train.Saver(max_to_keep = int(FLAGS.max_steps / FLAGS.ckpt_freq))

            #not sure if we need these yet
            #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.Session() as self.session:
                self.summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir,self.RUNID),
                                        self.session.graph)

                # Run the Op to initialize the variables.
                prev_steps =0
                if FLAGS.load is not None:
                    prev_steps = self.load_from_checkpoint()
                else:
                    self.session.run([init])

                #run
                #regardless of the algorithm, each step is one batch of data running through the generator
                #(or 2 if doing L1 with notre dame as well)
                for step in xrange(prev_steps, prev_steps + FLAGS.max_steps):

                    #get the new batch data
                    unsuper_feed_dict=None
                    super_feed_dict=None
                    if (FLAGS.gan and not FLAGS.use_nd_gan) or (FLAGS.L1 and not FLAGS.use_nd_L1):
                        unsuper_feed_dict = self.get_feed_dict(FLAGS.use_replay)
                    if FLAGS.use_nd_gan or FLAGS.use_nd_L1:
                        super_feed_dict = self.get_nd_feed_dict()

                    #choose which feed_dict to use for each component
                    if FLAGS.gan:
                        if FLAGS.use_nd_gan:
                            gan_feed_dict = super_feed_dict
                        else:
                            gan_feed_dict = unsuper_feed_dict

                    if FLAGS.L1:
                        if FLAGS.use_nd_L1:
                            l1_feed_dict = super_feed_dict
                        else:
                            l1_feed_dict = unsuper_feed_dict

                    if FLAGS.use_recon:
                        recon_feed_dict = unsuper_feed_dict

                    #run the operations
                    if FLAGS.L1:
                        self.l1(l1_fetches,l1_feed_dict)

                    if FLAGS.use_recon:
                        self.recon(recon_fetches,recon_feed_dict)

                    if FLAGS.gan:
                        if FLAGS.wgan:
                            self.wgan(d_fetches,g_fetches,gan_feed_dict,step)
                        else:
                            self.basic_gan(d_fetches,g_fetches,gan_feed_dict)

                    #push generated images to the cache if necessary
                    if FLAGS.use_replay:
                        self.replay.push(self.generated)

                    #process summaries, checkpoints, etc
                    if step % FLAGS.display_freq == 0:
                        self.display(step)
                        self.add_summaries(step,super_feed_dict,unsuper_feed_dict)

                    if (step + 1) % FLAGS.ckpt_freq == 0 or (step + 1) == FLAGS.max_steps:
                        self.checkpoint(step)

                    if (step + 1) % FLAGS.gen_print_freq == 0 or (step + 1) == FLAGS.max_steps:
                        if FLAGS.gan:
                            self.print_generated(gan_feed_dict[self.synth_placeholder],step)
                        elif FLAGS.L1:
                            self.print_generated(l1_feed_dict[self.synth_placeholder],step)

def main(_):
    training = Train()

    #create log directory
    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)
    tf.gfile.MakeDirs(os.path.join(FLAGS.log_dir,training.RUNID))

    #create cache directory
    if FLAGS.cache_path is not None:
        os.makedirs(os.path.dirname(FLAGS.cache_path), exist_ok=True)

    #write the command line arguments to a file
    flags_file = open(os.path.join(FLAGS.log_dir,training.RUNID,"hyperparams.txt"),"w")
    flags_file.write(" ".join(sys.argv[1:]))
    flags_file.close()

    #run
    training.run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--synth_dir',
        type=str,
        default='../Data/Faces_depth',
        help='Directory of Synthetic Images'
    )
    parser.add_argument(
        '--real_dir',
        type=str,
        default='../Data/Faces_lfwcropped',
        help='Directory of Real Images'
    )
    parser.add_argument(
        '--nd_dir',
        type=str,
        default='../Data/Faces_ND',
        help='Directory of Notre Dame Supervised Images'
    )
    parser.add_argument(
        '--use_nd_L1',
        action='store_true',
        default=False,
        help='use the supervised data set for pre-training'
    )
    parser.add_argument(
        '--use_nd_gan',
        action='store_true',
        default=False,
        help='use the supervised data set for pre-training'
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
        '--glr',
        type=float,
        default=1e-3,
        help='Generator Learning Rate'
    )

    parser.add_argument(
        '--dlr',
        type=float,
        default=1e-4,
        help='Discriminator learning rate.'
    )

    parser.add_argument(
        '--rlr',
        type=float,
        default=1e-6,
        help='Reconstruction learning rate'
    )

    parser.add_argument(
        '--llr',
        type=float,
        default=1e-3,
        help='L1 learning rate'
    )

    parser.add_argument(
        '--max_steps',
        type=int,
        default=10000,
        help='Number of steps to run trainer.'
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
        '--log_dir',
        type=str,
        default='logs',
        help='Directory to Save ckpts, summaries, and generated images'
        )
    parser.add_argument(
        '--channels',
        type=int,
        default=1,
        help='Number of channels in the input image'
        )
    parser.add_argument(
        '--optim',
        type=str,
        default="RMSP",
        help='Optimizer to use'
        )
    parser.add_argument(
        '--L1',
        action='store_true',
        default=False,
        help='Use L1 loss'
        )
    parser.add_argument(
        '--L1_weight',
        type=float,
        default=1,
        help='Weight given to L1 loss'
        )
    parser.add_argument(
        '--no_gan',
        action='store_false',
        dest='gan',
        help='Dont use a gan - just run l1 loss'
    )
    parser.add_argument(
        '--L2gan',
        action='store_true',
        default=False,
        help='Use L2 Discirminator Loss'
    )
    parser.add_argument(
        '--wgan',
        action='store_true',
        default=False,
        help='Use Wgan Approach'
    )

    parser.add_argument(
        '--diter',
        type=int,
        default=5,
        help='number of times to run the discriminator each pass for WGAN'
    )
    parser.add_argument(
        '--diff_low',
        type=float,
        default=.05,
        help='Require the discriminator to detect a difference of this much before switching training modes'
    )
    parser.add_argument(
        '--diff_high',
        type=float,
        default=.95,
        help='Dont run the discriminator if it is predicting with this level of accuracy'
    )
    parser.add_argument(
        '--display_freq',
        type=int,
        default=20,
        help='Frequency to print loss to stdout and to get summary values'
    )
    parser.add_argument(
        '--gen_print_freq',
        type=int,
        default= 500,
        help='Frequency at which to generate the images'
    )
    parser.add_argument(
        '--gen_num',
        type=int,
        default= 10,
        help='Number of Images to Print at every update'
    )
    parser.add_argument(
        '--ckpt_freq',
        type=int,
        default=1000,
        help='Frequency at which to save checkpoints'
    )
    parser.add_argument(
        '--batch_norm',
        action='store_true',
        default=False,
        help='Use batch norm for all layers'
    )
    parser.add_argument(
        '--use_replay',
        action='store_true',
        default=False,
        help='Use experience replay'
    )
    parser.add_argument(
        '--cache_path',
        type=str,
        default=None,
        help='If not none memory map the experience memory cache to this file'
    )
    parser.add_argument(
        '--cache_size',
        type=int,
        default=100000,
        help='number of examples to keep in the replay cache'
    )
    parser.add_argument(
        '--reuse_replay',
        action='store_true',
        default=False,
        help='whether to load the cache or start anew'
    )
    parser.add_argument(
        '--use_recon',
        action='store_true',
        default=False,
        help='Add Reconstruction Loss to the generator'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
