import tensorflow as tf

def LReLU(x,alpha):
    return tf.maximum(alpha*x,x)

class GanModel():
    def __init__(self,batch_size=10,image_size=128,gen_arch="U",batch_norm=False,noise_size=100):
        self.batch_size=batch_size
        self.image_size=image_size
        self.noise_size=noise_size

        if gen_arch == "U":
            self.generator = self.u_generator
        elif gen_arch == "V":
            self.generator = self.v_generator
        elif gen_arch == "F":
            self.generator = self.f_generator
        elif gen_arch == "NU":
            self.generator = self. u_generator_noise

        self.batch_norm=batch_norm
        print(self.batch_norm)

    def conv2d(self,x, input_channels, num_filters, f_size,strides=(1,1,1,1),activation=LReLU,use_bias=True,
               scope=None,training=True):
        with tf.variable_scope(scope):
            filters = tf.get_variable("filters",[f_size[0],f_size[1],input_channels, num_filters])
            result = tf.nn.conv2d(x, filters, strides, padding="SAME")

            if use_bias:
                bias = tf.get_variable("bias", [num_filters], initializer=tf.constant_initializer(0.0))
                result += bias

            if self.batch_norm:
                result = tf.contrib.layers.batch_norm(result,decay=.9,center=True,scale=True,
                                                      updates_collections=None,is_training=training)

            if activation == LReLU:
                alpha = tf.constant(.01)
                result = LReLU(result,alpha)
            else:
                result = activation(result)

            return result

    def deconv2d(self,x,image_size,input_channels, num_filters, f_size,strides=(1,1,1,1),
                 activation=LReLU,use_bias=True,scope=None,training=True):
        with tf.variable_scope(scope):

            filters = tf.get_variable("filters",[f_size[0],f_size[1],num_filters,input_channels])
            output_shape =[self.batch_size,int(image_size),int(image_size),num_filters]
            result = tf.nn.conv2d_transpose(x, filters, output_shape,strides, padding="SAME")

            if use_bias:
                bias = tf.get_variable("bias", [num_filters], initializer=tf.constant_initializer(0.0))
                result += bias

            if self.batch_norm:
                result = tf.contrib.layers.batch_norm(result,decay=.9,center=True,scale=True,
                    updates_collections=None, is_training=training)

            if activation == LReLU:
                alpha = tf.constant(.01)
                result = LReLU(result,alpha)

            else:
                result = activation(result)

            #if batch norm is true
            return result

    def dense(self,x, input_size, output_size, activation=None,use_bias=True,scope=None,training=True):
        with tf.variable_scope(scope):
            weights = tf.get_variable("weights",[input_size,output_size])
            result = tf.matmul(x,weights)

            if use_bias:
                bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))
                result += bias

            if self.batch_norm:
                result = tf.contrib.layers.batch_norm(result,decay=.9,center=True,scale=True,
                                                      updates_collections=None,is_training=training)

            if activation is not None:
                result = activation(result)

            return result

    def discriminator(self,images,input_channels,l2_gan=False,wgan=False):
        conv1 = self.conv2d(images,input_channels,32,[5, 5],[1,2,2,1],scope="conv1")
        conv2 = self.conv2d(conv1,32,32,[5, 5],[1,2,2,1],scope="conv2")
        conv3 = self.conv2d(conv2,32,16,[3, 3],scope="conv3")
        resized = tf.reshape(conv3, shape=[self.batch_size,int(self.image_size * self.image_size)])

        if l2_gan or wgan:
            pred = self.dense(resized,int(self.image_size * self.image_size),1,scope="dense")
        else:
            pred = self.dense(resized,int(self.image_size * self.image_size),1,
                              activation=tf.nn.sigmoid,scope="dense")

        return pred

    def f_generator(self,images,input_channels):
        """
        Flat generator - image remains the same size throughout
        """
        conv1 = self.conv2d(images,input_channels,32,[3, 3],scope="conv1")
        conv2 = self.conv2d(conv1,32,32,[3, 3],scope="conv2")
        conv3 = self.conv2d(conv2,32,32,[3, 3],scope="conv3")
        conv4 = self.conv2d(conv3,32,32,[3, 3],scope="conv4")
        conv5 = self.conv2d(conv4,32,input_channels,[1, 1],scope="conv5",activation=tf.nn.tanh)

        return conv5

    def u_generator(self,images,input_channels):
        """
        u-shaped architecture with skip connections
        """
        #encoder
        conv1 = self.conv2d(images,input_channels,32,[1,1],scope="conv1")
        conv2 = self.conv2d(conv1,32,32,[5, 5],[1,2,2,1],scope="conv2")
        conv3 = self.conv2d(conv2,32,32,[5, 5],[1,2,2,1],scope="conv3")

        #bottleneck
        conv4 = self.conv2d(conv3,32,32,[3, 3],scope="conv4")

        #decoder
        deconv1 = self.deconv2d(conv4,self.image_size/2,32,32,[5, 5],[1,2,2,1],scope="deconv1")
        d1_skip = tf.concat([conv2,deconv1],axis=3)
        deconv2 = self.deconv2d(d1_skip,self.image_size,64,32,[5, 5],[1,2,2,1],scope="deconv2")
        d2_skip = tf.concat([conv1,deconv2],axis=3)

        conv5 = self.conv2d(d2_skip,64,input_channels,[1, 1],scope="conv5",activation=tf.nn.tanh)

        return conv5

    def u_generator_noise(self,images,input_channels):
        """
        same architecture as u_generator, but add a dense layer
        connecting the bottleneck to a random noise vector
        """

        #encoder
        conv1 = self.conv2d(images,input_channels,32,[1,1],scope="conv1")
        conv2 = self.conv2d(conv1,32,32,[5, 5],[1,2,2,1],scope="conv2")
        conv3 = self.conv2d(conv2,32,32,[5, 5],[1,2,2,1],scope="conv3")

        #add a convolution to make this more reasonable in size
        conv4 = self.conv2d(conv3,32,1,[1,1],scope="conv4")
        flat = tf.reshape(conv4,shape=[self.batch_size,int(self.image_size * self.image_size / 16)])

        noise = tf.random_normal([self.batch_size,self.noise_size], mean=0, stddev=1)
        noise_added = tf.concat([noise,flat],axis=1)
        noise_dense = self.dense(noise_added,int(self.image_size * self.image_size / 16) + self.noise_size,
                                 int(self.image_size * self.image_size / 16),scope="noise_dense",
                                 activation=tf.nn.relu)
        noise_reshaped = tf.reshape(noise_dense,[self.batch_size,int(self.image_size/4),int(self.image_size/4),1])

        #create more filters again
        conv5 = self.conv2d(noise_reshaped,1,32,[1,1],scope="conv5")
        conv6 = self.conv2d(conv5,32,32,[3, 3],scope="conv6")

        #decoder
        deconv1 = self.deconv2d(conv6,self.image_size/2,32,32,[5, 5],[1,2,2,1],scope="deconv1")
        d1_skip = tf.concat([conv2,deconv1],axis=3)
        deconv2 = self.deconv2d(d1_skip,self.image_size,64,32,[5, 5],[1,2,2,1],scope="deconv2")
        d2_skip = tf.concat([conv1,deconv2],axis=3)

        conv7 = self.conv2d(d2_skip,64,input_channels,[1, 1],scope="conv7",activation=tf.nn.tanh)

        return conv7

    def v_generator(self,images,input_channels):
        """
        same architecture as u-generator, without skip connections
        """

        conv1 = self.conv2d(images,input_channels,32,[5, 5],[1,2,2,1],scope="conv1")
        conv2 = self.conv2d(conv1,32,32,[5, 5],[1,2,2,1],scope="conv2")
        conv3 = self.conv2d(conv2,32,32,[3, 3],scope="conv3")

        deconv1 = self.deconv2d(conv3,self.image_size/2,32,32,[5, 5],[1,2,2,1],scope="deconv1")
        deconv2 = self.deconv2d(deconv1,self.image_size,32,32,[5, 5],[1,2,2,1],scope="deconv2")

        conv4 = self.conv2d(deconv2,32,input_channels,[1, 1],scope="conv4",activation=tf.nn.tanh)

        return conv4

    def recon_g(self,synth_images,input_channels):
        with tf.variable_scope("G") as scope:
            generated_images = self.generator(synth_images,input_channels)
        with tf.variable_scope("R") as scope:
            recon_images = self.generator(generated_images,input_channels)

        r_loss = tf.reduce_mean(tf.abs(recon_images-synth_images))
        tf.summary.scalar("R",r_loss)
        return r_loss,recon_images

    def l1_g(self,synth_images,real_images,input_channels):
        with tf.variable_scope("G") as scope:
            generated_images = self.generator(synth_images,input_channels)
            l1_loss = tf.reduce_mean(tf.abs(real_images - generated_images))

        tf.summary.scalar("L1",l1_loss)
        return l1_loss,generated_images

    def combine_replay(self,generated_images,replay_images,input_channels):
        chosen_generated = tf.slice(generated_images,[0,0,0,0],
                [int(self.batch_size/2),self.image_size,self.image_size,input_channels])
        return tf.concat([chosen_generated,replay_images],axis=0)

    def gan(self,synth_images,real_images,input_channels,l2_gan=False,wgan=False,replay_images=None):
        """
        generative adversarial network
        if L2 gan is true, use L2 loss
        if wgan is true use wasserstain gan loss
        if replay_images is not none, use experience replay mechanism
        """
        with tf.variable_scope("G") as scope:
            try:
                generated_images = self.generator(synth_images,input_channels)
            except:
                scope.reuse_variables()
                generated_images = self.generator(synth_images,input_channels)

        with tf.variable_scope('D') as scope:
            fake_images = generated_images if replay_images is None \
                          else self.combine_replay(generated_images,replay_images,input_channels)
            try:
                synth_preds = self.discriminator(fake_images,input_channels,l2_gan)
            except:
                scope.reuse_variables()
                synth_preds = self.discriminator(fake_images,input_channels,l2_gan)

            scope.reuse_variables()
            real_preds = self.discriminator(real_images,input_channels,l2_gan)

            difference = tf.reduce_mean(real_preds-synth_preds)


        if l2_gan:
            d_loss = tf.reduce_mean((1-real_preds)**2 + synth_preds**2)
            g_loss = tf.reduce_mean((1-synth_preds)**2)

        elif wgan:
            d_loss = tf.reduce_mean(real_preds - synth_preds)
            g_loss = tf.reduce_mean(synth_preds)

        else:
            d_loss = -tf.reduce_mean(tf.log(real_preds + 1e-7) + tf.log((1-synth_preds) + 1e-7))
            g_loss = -tf.reduce_mean(tf.log(synth_preds + 1e-7))

        to_return = [d_loss,g_loss,generated_images,real_preds,synth_preds,difference]

        if wgan:
            d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="D")
            clip_d = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]
            to_return.append(clip_d)

        tf.summary.scalar("D", d_loss)
        tf.summary.scalar("G", g_loss)
        tf.summary.scalar("Difference", difference)

        return to_return
