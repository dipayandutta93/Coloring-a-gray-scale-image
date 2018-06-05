from preprocessor import loadData
import tensorflow as tf
from model import model
from preprocessor import batch_iter,load_variable
# from sklearn.preprocessing import normalize
import numpy as np
train_data,train_labels,test_data,test_labels = load_variable()
    # loadData(path1=r'C:\Users\pravi\Downloads\input.csv',path2=r'C:\Users\pravi\Downloads\color.csv')
channel = 2
train_labels = train_labels[:,channel,:]
test_labels = test_labels[:,channel,:]
print("data loaded....")
batch_size = 4096
m = model(batch_size=batch_size,input_dim=9,output_dim=256)
learning_rate = 1e-3
epochs = 10000
# Define Training procedure
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
grads_and_vars = optimizer.compute_gradients(m.loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

session_conf = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)

sess = tf.Session(config=session_conf)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
train_batches = batch_iter(list(zip(train_data,train_labels)),batch_size=batch_size,epochs=epochs,Isshuffle=False)
i = 0
best_loss_dev = 100000
for batch in train_batches:
    x,y  = zip(*batch)
    x = np.array(x)
    y = np.array(y)
    feed_dict = {
        m.x : x,
        m.y : y,
        m.drop_out_keep_prob:0.8
    }
    _,loss = sess.run([train_op,m.loss],feed_dict=feed_dict)
    print("step - " + str(i) + "    loss is " + str(loss))

    if i%50 == 0 and i > 0:
        j = 0
        test_loss = 0.0
        test_batches = batch_iter(list(zip(test_data,test_labels)), batch_size=batch_size, epochs=1,Isshuffle=False)
        for test_batch in test_batches:
            x,y = zip(*test_batch)
            x = np.array(x)
            y = np.array(y)
            feed_dict = {
                m.x: x,
                m.y: y,
                m.drop_out_keep_prob:1.0
            }
            loss = sess.run(m.loss, feed_dict=feed_dict)
            test_loss += loss
            j += 1
        print(" test loss is " + str(test_loss/j))
        if test_loss/j < best_loss_dev:
            best_loss_dev = test_loss/j
            save_path = "saved_models_"+str(channel)+"/model-" + str(i)
            saver.save(sess, save_path=save_path)
            print("Model saved to " + save_path)
    i += 1
print("test-loss "+str(best_loss_dev))

