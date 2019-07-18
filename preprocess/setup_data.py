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


def main(args):
	schema = pd.read_csv(args.schema_dir)

	#check to see if dirs exist
	if not os.path.exists(args.train_dir):
		os.mkdir(args.train_dir)

	if not os.path.exists(args.test_dir):
		os.mkdir(args.test_dir)
	
	#set up some variables
	save_ext = '.png'
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

			try:
				#create new file
				number = ("%04d"%style)[:4]
				new_fpath = os.path.join(fam_path, 'fam' + str(family) + '_' + number +'.png')
				img = Image.open(file)
				img.save(new_fpath, 'png')
				#print(fname+" to "+new_fpath)
			except:
				print('error: ' + fname)


if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))