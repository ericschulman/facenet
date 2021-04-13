import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import argparse, sys, glob, os
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.python.ops import data_flow_ops
import validate_on_lfw
import lfw
import facenet

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--schema_dir', type=str, 
        help='Directory with schema.', 
        default='../datasets/UT research project datasets/Style Sku Family.csv')
    parser.add_argument('--model', type=str, 
        help='Directory with preprocessed data.', 
        default='../models/20210407-203650')
    parser.add_argument('--log', type=str, 
        help='Directory with preprocessed data.', 
        default='../logs/20210407-203650')
    parser.add_argument('--data_dir', type=str, 
        help='Directory with data.', 
        #default='../datasets/npg_small/')
        default='../datasets/New Pangram 2/')
    parser.add_argument('--write_dir', type=str, 
        help='Directory with data.', 
        default='../datasets/UT research project datasets/')
    parser.add_argument('--train_dir', type=str, 
        help='Directory with data.', 
        default='../datasets/crop7_train')
    parser.add_argument('--test_dir', type=str, 
        help='Directory with data.', 
        default='../datasets/crop7_test')
    return parser.parse_args(argv)



def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
        embeddings, labels, image_paths, batch_size):

    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')

    # Enqueue one epoch of image paths and labels
    nrof_images = len(image_paths)
    labels_array = np.expand_dims(np.arange(0,nrof_images),1)
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths),1),1)
    control_array = np.zeros_like(labels_array, np.int32)
    control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION

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
            #print('.', end='')
            sys.stdout.flush()
    print('')
    embeddings = np.zeros((nrof_images, embedding_size))
    embeddings = emb_array

    return embeddings




def get_embeddings(image_paths, model, batch_size):
    embeddings = None
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            paths = image_paths
            actual_issame = None
            
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
 
            nrof_preprocess_threads = 4
            image_size = (100, 100)
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                        dtypes=[tf.string, tf.int32, tf.int32],
                                        shapes=[(1,), (1,), (1,)],
                                        shared_name=None, name=None)
            eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')
            image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)
     
            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(model, input_map=input_map)

            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")     
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)

            #for the first round save the input map, so that we can visualize images of positives
            embeddings = evaluate(sess, eval_enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
                embeddings, label_batch, paths, batch_size)
    return embeddings


def find_images(args):
    schema = pd.read_csv(args.schema_dir)

    #set up df with information
    n = 128
    embedding_names = ['embedding %s'%(i+1) for i in range(n)]
    result_df = pd.DataFrame(columns=['img_name','style','family','crop_name']+embedding_names)
    
    #set up some variables
    img_files = glob.glob(args.data_dir + '*.bmp')
    cropped_images = []

    for file in img_files:
        #process image name
        fpath, fname = os.path.split(file)
        style = int(fname.split('=')[1].split('.')[0])

        #process family folder
        family = schema[schema['Style ID']==style]
        result_df = result_df.append({'img_name':fname,'style':style}, ignore_index=True)
        if not family.empty:
            family = family['Family ID'].iloc[0]
            result_df['family'][result_df['img_name'] == fname] = family

            number = ("%03d"%style)[:3]   
            fam_path = os.path.join(args.train_dir , 'fam' + str(family))
            img_name = 'fam' + str(family) + '_' + (number)

            img_name = fam_path + '/' + img_name
            img_names = glob.glob(img_name + '*.png')
            if len(img_names) > 0:
                cropped_images.append(img_names[0])
                result_df['crop_name'][result_df['img_name'] == fname] = img_names[0]

    result_df.to_csv(args.write_dir + '/embeddings_full.csv',index=False, header=True)


def write_embeddings(args):
	result_df = pd.read_csv(args.write_dir + '/embeddings_full.csv')
	result_df = result_df[['img_name','style','family','crop_name']]
	cropped_images = result_df['crop_name'].dropna()
	cropped_images = list(cropped_images)
	batch_size = 50
	end = len(cropped_images)%batch_size

	embeddings = get_embeddings(cropped_images[:-end],args.model,batch_size)

	embeddings_names = ['embedding %s'%(i+1) for i in range(128)]
	embeddings_df = pd.DataFrame(data=embeddings, columns=embeddings_names)
	embeddings_df['crop_name'] = cropped_images[:-end]

	result_df = result_df.merge(embeddings_df, on=(['crop_name']), how='left')
	result_df.to_csv(args.write_dir + '/embeddings_full.csv',index=False, header=True)


if __name__ == '__main__':
    find_images(parse_arguments(sys.argv[1:]))
    write_embeddings(parse_arguments(sys.argv[1:]))