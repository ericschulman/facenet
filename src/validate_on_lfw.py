"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
from tensorflow.python.ops import data_flow_ops
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.contrib.tensorboard.plugins import projector


def main(args):
    embeddings, image_paths = None, None
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)
            
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
 
            nrof_preprocess_threads = 4
            image_size = (args.image_size, args.image_size)
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                        dtypes=[tf.string, tf.int32, tf.int32],
                                        shapes=[(1,), (1,), (1,)],
                                        shared_name=None, name=None)
            eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')
            image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)
     
            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(args.model, input_map=input_map)

            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")     
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)

            #for the first round save the input map, so that we can visualize images of positives
            embeddings, image_paths = evaluate(sess, eval_enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
                embeddings, label_batch, paths, actual_issame, args.lfw_batch_size, args.lfw_nrof_folds, args.distance_metric, args.subtract_mean,
                args.use_flipped_images, args.use_fixed_image_standardization, logdir = args.logs)

    #save the examples to an event file
    ####################################################################################################
    #TODO: uncomment this stuff ########################################################################
    ####################################################################################################

    #save_examples(args.logs)
    #save_embeddings(embeddings, image_paths, args.logs)



def save_embeddings(embeddings,filenames, logdir):
    no_images, no_embs = embeddings.shape
    #open and save images...
    image_ph = tf.placeholder(tf.float32, [None, 100, 100, 3])
    #random_images = np.random.uniform(0.0, 1.0, (no_images, 100, 100, 3))
    thumbnails = []
    for file in filenames:
        img = mpimg.imread(file)
        thumbnails.append(img)

    #random_labels = np.random.randint(0, 1, (no_images, ))
    thumbnails = np.array(thumbnails)
    # Embedding variable that stores the embeddings
    embedding_var = tf.Variable(embeddings)
    
    # Create embedding projector, specify embedding inputs to the projector
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Specify sprite images ---> The images corresponding to the embeddings you are computing
    embedding.sprite.image_path = 'sprite_images.png'
    embedding.sprite.single_image_dim.extend([100, 100])
    # Create a summary file writer
    writer = tf.summary.FileWriter(logdir)
    # Add the embedding visualizations to this summary
    projector.visualize_embeddings(writer, config)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(embedding_var, feed_dict={image_ph: thumbnails})
        # Save the computed embeddigns
        saver = tf.train.Saver()
        saver.save(sess, logdir + "/model.ckpt", 0)
        
    write_sprite_image(logdir + '/sprite_images.png', thumbnails[..., 0])


def write_sprite_image(filename, images):
    """ Create a sprite image consisting of sample images
        :param filename: name of the file to save on disk
        :param shape: tensor of flattened images  """
    img_h = images.shape[1]
    img_w = images.shape[2]
    # Calculate number of plot
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    # Make the background of sprite image
    sprite_image = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            img_idx = i * n_plots + j
            if img_idx < images.shape[0]:
                img = images[img_idx]
                sprite_image[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = img

    plt.imsave(filename, sprite_image, cmap='gray')
    print('Sprite image saved in {}'.format(filename))
         

def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
        embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, distance_metric, subtract_mean, use_flipped_images, use_fixed_image_standardization, logdir=None):
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')

    # Enqueue one epoch of image paths and labels
    nrof_embeddings = len(actual_issame)*2  # nrof_pairs * nrof_images_per_pair
    nrof_flips = 2 if use_flipped_images else 1
    nrof_images = nrof_embeddings * nrof_flips
    labels_array = np.expand_dims(np.arange(0,nrof_images),1)
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths),nrof_flips),1)
    control_array = np.zeros_like(labels_array, np.int32)
    if use_fixed_image_standardization:
        control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION
    if use_flipped_images:
        # Flip every second image
        control_array += (labels_array % 2)*facenet.FLIP
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    
    embedding_size = int(embeddings.get_shape()[1])
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    embeddings = np.zeros((nrof_embeddings, embedding_size*nrof_flips))
    if use_flipped_images:
        # Concatenate embeddings for flipped and non flipped version of the images
        embeddings[:,:embedding_size] = emb_array[0::2,:]
        embeddings[:,embedding_size:] = emb_array[1::2,:]
    else:
        embeddings = emb_array

    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'

    tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(embeddings, actual_issame, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean, labels=image_paths, 
        logdir= logdir)
    
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    
    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)
    return embeddings, image_paths
   

def save_examples(logdir = None):
    """save examples of true and false positives to help 
    visualize the learning"""
    if logdir is None:
        return

    input_map = {}

    for error_type in ['false_negatives', 'false_positives','true_positives', 'true_negatives']:
        #load the names of the files
        files = np.loadtxt(logdir + '/' + error_type + '.csv', dtype=str, delimiter = ',')
        if len(files.shape)  < 2:
            files = np.array([files])

        files1 = files[:,0]
        files2 = files[:,1]

        wrong_pairs = np.zeros((1,0,200,3)) #create a null image to concat to

        #for each pair...
        for i in range(len(files1)):
            wrong_row = np.zeros((1,100,0,3))
            #cycle through both types of files
            for j in range(2) :
                filelist = [files1,files2][j]
                image_paths_placeholder = tf.placeholder(tf.string, name='image_paths'+str(i) +str(j))
                input_map[image_paths_placeholder] = filelist[i]
                #read the contents of the file and write it
                file_contents = tf.read_file(image_paths_placeholder)
                img = tf.image.decode_image(file_contents)
                img = tf.reshape(img, (1,100,100,3)) #TODO: hard coded dimensions
                wrong_row = tf.concat((wrong_row,img),axis=2)
            wrong_pairs = tf.concat((wrong_pairs,wrong_row),axis=1)
        
        #concat row to total
        tf.summary.image(error_type, wrong_pairs, max_outputs=100)

    #run a small network just to save the output    
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        summary = sess.run(summary_op, feed_dict=input_map)
        writer = tf.summary.FileWriter(logdir)
        writer.add_summary(summary, 0)




    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--distance_metric', type=int,
        help='Distance metric  0:euclidian, 1:cosine similarity.', default=0)
    parser.add_argument('--use_flipped_images', 
        help='Concatenates embeddings for the image and its horizontally flipped counterpart.', action='store_true')
    parser.add_argument('--subtract_mean', 
        help='Subtract feature mean before calculating distance.', action='store_true')
    parser.add_argument('--use_fixed_image_standardization', 
        help='Performs fixed standardization of images.', action='store_true')
    parser.add_argument('--logs', 
        help='Log directory for saving visualizations.', default=None)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
