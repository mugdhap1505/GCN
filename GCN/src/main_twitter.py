"""
Created on Sat Mar 28 21:52:08 2020

@author: mugdh
"""

import time
import scipy.sparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tensorflow as tf
import graph as lg
import utils as us
import sys
# features, adj, labels, edges = us.load_dataset()
# y_train, y_val, train_mask, val_mask = us.get_splits_dataset(labels)
# g = nx.read_graphml('../data/twitter-graph.graphml')
g = nx.read_edgelist('../data/edge_list_new.txt', delimiter = '\t')

print('Dataset has {} nodes, {} edges.'.format(g.number_of_nodes(), g.number_of_edges()))
 # nx.draw(
 #     g,
 #     cmap=plt.get_cmap('jet'),
 #     node_color=np.log(list(nx.get_node_attributes(g, 'membership').values())))
 
adj = nx.adj_matrix(g)

# print(adj.shape)
# sys.exit()
 
#Get important parameters of adjacency matrix
n_nodes = g.number_of_nodes()
# Some preprocessing
adj_tilde = adj + np.identity(n=adj.shape[0])
d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))
d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1/2)
d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)
adj_norm_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(adj))

#Features are just the identity matrix
feat_x = np.identity(n=adj.shape[0])
feat_x_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(feat_x))
#Semi-supervised
idx_features_labels = np.genfromtxt('../data/clustering_new.txt', dtype=np.dtype(int))
class_lables = np.array(idx_features_labels[:, 1], dtype = int)
one_hot_targets = us.encode_onehot(class_lables)
# memberships = [m - 1
#for m in nx.get_node_attributes(g, 'membership').values()]
#print("memebership:", memberships)
nb_classes = one_hot_targets.shape[1]
# targets = np.array([memberships], dtype=np.int32).reshape(-1)
# # print("targets:", targets)
# one_hot_targets = np.eye(nb_classes)[targets]
# Pick one at random from each class
# labels_to_keep = [np.random.choice(
#     np.nonzero(one_hot_targets[:, c])[0]) for c in range(nb_classes)]
# print("labels to keep:", labels_to_keep)
nodes_per_class = 50
#for i in range(nodes_per_class):
labels_to_keep0 = [np.random.choice(np.nonzero(one_hot_targets[:, 0])[0]) for c in range(nodes_per_class)]
labels_to_keep1 = [np.random.choice(np.nonzero(one_hot_targets[:, 1])[0]) for c in range(nodes_per_class)]
labels_to_keep = labels_to_keep0 + labels_to_keep1

y_train = np.zeros(shape=one_hot_targets.shape,dtype=np.float32)
y_val = one_hot_targets.copy()

train_mask = np.zeros(shape=(n_nodes,), dtype=np.bool)
val_mask = np.ones(shape=(n_nodes,), dtype=np.bool)

for l in labels_to_keep:
    y_train[l, :] = one_hot_targets[l, :]
    y_val[l, :] = np.zeros(shape=(nb_classes,))
    train_mask[l] = True
    val_mask[l] = False
# print("y_train:", y_train)
# print("train_mask:", train_mask)
# print("y_val:", y_val)
# print("val_mask:", val_mask)
# nb_classes = labels.shape[1]
# n_nodes = labels.shape[0]
# TensorFlow placeholders
ph = {
    'adj_norm': tf.sparse_placeholder(tf.float32, name="adj_mat"),
    'x': tf.sparse_placeholder(tf.float32, name="x"),
    'labels': tf.placeholder(tf.float32, shape=(n_nodes, nb_classes)),
    'mask': tf.placeholder(tf.int32)}

l_sizes = [16, 4, 2, nb_classes]

o_fc1 = lg.GraphConvLayer(input_dim=feat_x.shape[-1],
                          output_dim=l_sizes[0],
                          name='fc1',
                          act=tf.nn.tanh)(adj_norm=ph['adj_norm'],
                                          x=ph['x'], sparse=True)

o_fc2 = lg.GraphConvLayer(input_dim=l_sizes[0],
                          output_dim=l_sizes[1],
                          name='fc2',
                          act=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc1)

o_fc3 = lg.GraphConvLayer(input_dim=l_sizes[1],
                          output_dim=l_sizes[2],
                          name='fc3',
                          act=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc2)

o_fc4 = lg.GraphConvLayer(input_dim=l_sizes[2],
                          output_dim=l_sizes[3],
                          name='fc4',
                          act=tf.nn.softmax)(adj_norm=ph['adj_norm'], x=o_fc3)


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

with tf.name_scope('optimizer'):
    loss = masked_softmax_cross_entropy(
        preds=o_fc4, labels=ph['labels'], mask=ph['mask'])

    accuracy = masked_accuracy(
        preds=o_fc4, labels=ph['labels'], mask=ph['mask'])

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    opt_op = optimizer.minimize(loss)

feed_dict_train = {ph['adj_norm']: adj_norm_tuple,
                   ph['x']: feat_x_tuple,
                   ph['labels']: y_train,
                   ph['mask']: train_mask}

feed_dict_val = {ph['adj_norm']: adj_norm_tuple,
                 ph['x']: feat_x_tuple,
                 ph['labels']: y_val,
                 ph['mask']: val_mask}

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 50
save_every = 1

t = time.time()
outputs = {}
trainAccuracy = []
testAccuracy = []
trainLoss = []
testLoss = []
# Train model
for epoch in range(epochs):
    # Construct feed dictionary

    # Training step
    _, train_loss, train_acc = sess.run(
        (opt_op, loss, accuracy), feed_dict=feed_dict_train)
    trainAccuracy.append(train_acc)
    trainLoss.append(train_loss)
#     if epoch % save_every == 0:
        # Validation
    val_loss, val_acc = sess.run((loss, accuracy), feed_dict=feed_dict_val)
    testAccuracy.append(val_acc)
    testLoss.append(val_loss)
    # Print results
    #if epoch % save_every == 0:
    print("Epoch:", '%04d' % (epoch + 1),
      "train_loss=", "{:.5f}".format(train_loss),
      "train_acc=", "{:.5f}".format(train_acc),
      "val_loss=", "{:.5f}".format(val_loss),
      "val_acc=", "{:.5f}".format(val_acc),
      "time=", "{:.5f}".format(time.time() - t))

    feed_dict_output = {ph['adj_norm']: adj_norm_tuple,
                        ph['x']: feat_x_tuple}

    output = sess.run(o_fc3, feed_dict=feed_dict_output)
    outputs[epoch] = output

def PlotHistory(train,test,trainLoss,testLoss):
    #plot model loss and accuracy
    fig = plt.figure(figsize=(6,4))
    plt.plot(train)
    plt.plot(test)
    plt.title('Model Accuracy',fontsize = 10)
    plt.ylabel('Accuracy',fontsize = 10)
    plt.xlabel('Epochs',fontsize = 10)
    plt.legend(['train', 'test'], loc='upper left',fontsize = 10)
    fig.savefig("accuracy2.jpg")
    plt.show()
    # summarize history for loss
    fig = plt.figure(figsize=(7,4))
    plt.plot(trainLoss)
    plt.plot(testLoss)
    plt.title('Model Loss',fontsize = 10)
    plt.ylabel('Loss',fontsize = 10)
    plt.xlabel('Epochs',fontsize = 10)
    plt.legend(['train', 'test'], loc='upper left',fontsize = 10)
    fig.savefig("loss2.jpg")
    plt.show()


def saveResultsInFile(trainAccuracy, testAccuracy, trainLoss, testLoss):
    accuracyFile = open("../results2/accuracy2.csv",'a')
    lossFile = open("../results2/loss2.csv",'a')
    for i in range(len(trainAccuracy)):
        accuracyFile.write(str(trainAccuracy[i])+",")
        accuracyFile.write(str(testAccuracy[i]))
        accuracyFile.write('\n')
        lossFile.write(str(trainLoss[i])+",")
        lossFile.write(str(testLoss[i]))
        lossFile.write('\n')
    accuracyFile.close()
    lossFile.close()
#saveResultsInFile(trainAccuracy, testAccuracy,trainLoss, testLoss)  
PlotHistory(trainAccuracy,testAccuracy,trainLoss,testLoss)                          
