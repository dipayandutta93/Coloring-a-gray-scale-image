import tensorflow as tf

class model(object):

    def __init__(self,
                 batch_size,
                 input_dim,
                 output_dim
                 ):

        self.x = tf.placeholder(tf.float32,[batch_size,input_dim],name='x')
        self.y = tf.placeholder(tf.float32,[batch_size,output_dim],name='y')
        self.drop_out_keep_prob = tf.placeholder(tf.float32,name='drop_out_keep_prob')
        l2_loss = tf.constant(value=0.0, dtype=tf.float32)
        lmd = 1e-4
        layer_1_dim = 128
        layer_2_dim = 64
        layer_3_dim = 12
        ####################################L-1################################################
        W1 = tf.get_variable("W1",shape=[input_dim,layer_1_dim],
                                            initializer=tf.contrib.layers.xavier_initializer()
                                            )
        bias_1 = tf.Variable(initial_value=tf.constant(value=0.01,shape=[layer_1_dim]),name='bias_1')
        p = tf.matmul(self.x,W1) + bias_1
        layer1_out = tf.nn.sigmoid(p,name='layer1_out')
        layer1_out = tf.layers.dropout(layer1_out,rate=1.0 - self.drop_out_keep_prob)
        ###################################L-2#################################################
        W2 = tf.get_variable("W2", shape=[layer_1_dim, layer_2_dim],
                             initializer=tf.contrib.layers.xavier_initializer()
                             )

        bias_2 = tf.Variable(initial_value=tf.constant(value=0.01,shape=[layer_2_dim]),name='bias_2')
        p_2 = tf.matmul(layer1_out,W2) + bias_2
        layer2_out = tf.nn.sigmoid(p_2,name='layer2_out')
        layer2_out = tf.layers.dropout(layer2_out,rate=1.0 - self.drop_out_keep_prob)
        ####################################L-3################################################
        W3 = tf.get_variable("W3", shape=[layer_2_dim,layer_3_dim],
                             initializer=tf.contrib.layers.xavier_initializer()
                             )

        bias_3 = tf.Variable(initial_value=tf.constant(value=0.01,shape=[layer_3_dim]),name='bias_3')
        p_3 = tf.matmul(layer2_out,W3) + bias_3
        self.layer3_out = tf.sigmoid(p_3)
        self.layer3_out = tf.layers.dropout(self.layer3_out, rate=1.0 - self.drop_out_keep_prob)
        ########################################L-4##############################################

        W4 = tf.get_variable("W4", shape=[layer_3_dim, output_dim],
                             initializer=tf.contrib.layers.xavier_initializer()
                             )

        bias_4 = tf.Variable(initial_value=tf.constant(value=0.01, shape=[output_dim]), name='bias_4')
        self.p_4 = tf.matmul(self.layer3_out, W4) + bias_4
        self.pred = tf.argmax(self.p_4,1)
        ######################################################################################

        l2_loss += tf.nn.l2_loss(W1)
        l2_loss += tf.nn.l2_loss(bias_1)


        l2_loss += tf.nn.l2_loss(W2)
        l2_loss += tf.nn.l2_loss(bias_2)

        with tf.name_scope("loss-layer"):
            self.l = tf.nn.softmax_cross_entropy_with_logits(logits=self.p_4,labels=self.y)
            self.loss = tf.reduce_mean(self.l)
                        # + lmd * l2_loss
# model(batch_size=512,input_dim=9,output_dim=3)