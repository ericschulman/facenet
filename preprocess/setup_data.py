import argparse, sys, glob, os
import pandas as pd
import numpy as np
from PIL import Image


def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--schema_dir', type=str, 
		help='Directory with schema.', 
		default='../datasets/UT research project datasets/Style Sku Family.csv')
	parser.add_argument('--data_dir', type=str, 
		help='Directory with data.', 
		default='../datasets/npg_small/')
	parser.add_argument('--percent', type=float, 
		help='percentage of data to train with.', 
		default=.8)
	parser.add_argument('--test_dir', type=str, 
		help='Directory with data.', 
		default='../datasets/small_test/')
	parser.add_argument('--train_dir', type=str, 
		help='Directory with data.', 
		default='../datasets/small_train/')
	return parser.parse_args(argv)


def preprocess(img):
	lines = crop_sentences(img)
	words = []
	for line in lines:
		print('yo')
		raw_pix = np.array(line)
		start_index = 0
		while start_index < raw_pix.shape[1] :
			jump  = np.argmax( raw_pix[:,start_index:].min(axis=0) )
			print('start', start_index, jump, raw_pix.shape[1])
			if  jump > 0:
				print('update1', start_index, start_index + jump, raw_pix.shape[1])
				pix = raw_pix[:,start_index:start_index + jump]
				crop_img = Image.fromarray(pix)
				words.append(crop_img)
				start_index = jump + start_index
			else:
				start_index = start_index +1
				print('update2', start_index, start_index + jump, raw_pix.shape[1])
		#while we are before the end of the line
	return words


def crop_sentences(img):
	"""cut down on the size of the images by making smaller 100x100 chunks of 
	the panagram"""
	raw_pix = np.array(img)
	preprocess_im = []
	#divide the panagram into lines
	for pg_line in [0,1]:

		#divide the image into 2 halves
		pix = raw_pix[pg_line*200:200+pg_line*200, :]

		#find the corners
		topcol = np.argmax((np.argmax(pix!=255,axis=0) > 0))
		botcol = np.argmax((np.argmax(np.flip(pix,axis=(0,1))!=255,axis=0) > 0))
		toprow = np.argmax((np.argmax(pix!=255,axis=1) > 0))
		botrow = np.argmax((np.argmax(np.flip(pix,axis=(0,1))!=255,axis=1) > 0))

		pix= pix[toprow:200-botrow,topcol:400-botcol]
		crop_img = Image.fromarray(pix)
		preprocess_im.append(crop_img)

	return preprocess_im



def main(args):
	schema = pd.read_csv(args.schema_dir)

	#check to see if dirs exist
	if not os.path.exists(args.train_dir):
		os.mkdir(args.train_dir)

	if not os.path.exists(args.test_dir):
		os.mkdir(args.test_dir)
	
	#set up some variables
	img_files = glob.glob(args.data_dir + '*.bmp')
	for file in img_files:
		#process image name
		fpath, fname = os.path.split(file)
		style = int(fname.split('=')[1].split('.')[0])
		#process family folder
		family = schema[schema['Style ID']==style]

		if not family.empty:

			family = family['Family ID'].iloc[0]
			write_dir = np.random.choice(a=[args.train_dir, args.test_dir], p=[args.percent,1-args.percent])
			fam_path = os.path.join(write_dir , 'fam' + str(family))
			if not os.path.exists(fam_path):
				os.mkdir(fam_path)

			img = Image.open(file)
			number = ("%03d"%style)[:3]

			processed_imgs = preprocess(img)
			for p_ind in range(len(processed_imgs)):
				new_fpath = os.path.join(fam_path, 'fam' + str(family) + '_' + number + str(p_ind) +'.png')
				processed_imgs[p_ind].save(new_fpath, 'png')


if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))