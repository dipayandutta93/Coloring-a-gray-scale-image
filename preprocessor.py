import numpy as np
import random
import pickle

def save_variable(a,path):
    f = open(path,'wb')
    pickle.dump(a,f)
    f.close()

def load_variable(path = r'C:\Users\pravi\PycharmProjects\Assignment-4\data_pickles\data'):
    f = open(path,'rb')
    a = pickle.load(f)
    f.close()
    return a


def loadData(path1,path2):

    f= open(path1)
    lines = f.readlines()
    # random.shuffle(lines)
    data_count = len(lines)
    a = list(range(data_count))
    random.shuffle(a)
    ratio = 0.8
    train_data_size = int(ratio*data_count)
    test_data_size = data_count - train_data_size
    train_data = np.zeros([train_data_size,9],dtype=float)
    test_data = np.zeros([test_data_size,9],dtype=float)

    for i,index in enumerate(a):
        line = lines[index]
        vals = line.strip('\n').split(',')
        if i < train_data_size:
            for j in range(9):
                train_data[i,j] = int(vals[j])
        else:
            for j in range(9):
                test_data[i-train_data_size,j] = int(vals[j])
    f1 = open(path2)
    lines1 = f1.readlines()
    labels = np.zeros([data_count,3,256],dtype=float)
    count = np.zeros([256])
    for i,index in enumerate(a):
        line = lines1[index]
        vals = line.strip('\n').split(',')
        for j in range(3):
            labels[i,j,int(vals[j])] = 1
            if j ==1:
                count[int(vals[j])] += 1
    train_labels = labels[:train_data_size]
    test_labels = labels[train_data_size:]

    res = [train_data,train_labels,test_data,test_labels]
    save_variable(res,path=r'C:\Users\pravi\PycharmProjects\Assignment-4\data_pickles\data')

    return res

def batch_iter(data, batch_size, epochs, Isshuffle=True):
    ## check inputs
    assert isinstance(batch_size, int)
    assert isinstance(epochs, int)
    assert isinstance(Isshuffle, bool)

    num_batches = int((len(data) - 1) / batch_size)
    ## data padded
    # data = np.array(data + data[:2 * batch_size])
    data_size = len(data)
    print("size of data" + str(data_size) + "---" + str(len(data)))
    for ep in range(epochs):
        if Isshuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            yield shuffled_data[start_index:end_index]

# loadData(path1=r'C:\Users\pravi\Downloads\input.csv',path2=r'C:\Users\pravi\Downloads\color.csv')