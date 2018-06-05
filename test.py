from preprocessor import loadData
import tensorflow as tf
from model import model
from preprocessor import batch_iter,load_variable
# from sklearn.preprocessing import normalize
import numpy as np
# train_data,train_labels,test_data,test_labels = load_variable()
    # loadData(path1=r'C:\Users\pravi\Downloads\input.csv',path2=r'C:\Users\pravi\Downloads\color.csv')
channel = 1

f = open(r'C:\Users\pravi\Downloads\data.csv')
lines = f.readlines()
train_data = np.zeros([len(lines),9])
for i,line in enumerate(lines):
    splt = line.strip('\n').split(',')
    for j in range(9):
        train_data[i,j] = int(splt[j])
# train_labels = train_labels[:,channel,:]
# test_labels = test_labels[:,channel,:]
print("data loaded....")
batch_size = len(train_data)
m = model(batch_size=batch_size,input_dim=9,output_dim=256)
session_conf = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)

sess = tf.Session(config=session_conf)
saver = tf.train.Saver()

saver.restore(sess,r'C:\Users\pravi\PycharmProjects\Assignment-4\saved_models_'+str(channel)+'\model-37100')

feed_dict = {
        m.x : train_data,
        m.drop_out_keep_prob:1.0
    }
f = open(r'C:\Users\pravi\PycharmProjects\Assignment-4\output\out_channel_'+str(channel),'w')
predictions = sess.run(m.pred,feed_dict=feed_dict)
for i in range(len(predictions)):
    f.write(str(predictions[i]) + "\n")
# diff = abs(np.argmax(train_labels,axis=1) - predictions)
# mean_square_loss = np.mean(diff*diff)
f.close()
print("done")