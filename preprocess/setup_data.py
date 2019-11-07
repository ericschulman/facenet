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


def preprocess1(img):
	"""for now, break each sentence down further into individual words 
	using vertical white spaces"""
	lines = crop_sentences(img)
	words = []
	for line in lines:
		raw_pix = np.array(line)
		start_index = 0
		space = 0
		min_dist = max(2, 1 + int(raw_pix.shape[1]*.02))
		while start_index + space < raw_pix.shape[1] :
			jump  = np.argmax( (raw_pix[:,start_index + space:] ).min(axis=0) )
			if  jump > min_dist: #the necessary size of a jump before a new character
				pix = raw_pix[:,start_index:start_index + space + jump]
				crop_img = Image.fromarray(pix)
				words.append(crop_img)
				start_index = start_index +  jump + space
				space = 0
			else:
				space = space +1

	return words


def preprocess2(img):
	"""an alternate way to chop down the sentences"""
	lines = crop_sentences(img)
	
	top_breaks = np.array([0,6,11,15,20])/20
	bot_breaks =  np.array([0,4,8,14])/14
	all_breaks = [top_breaks,bot_breaks]
	
	words = []

	for i in [0,1]:
		raw_pix = np.array(lines[i])
		pix_breaks = (all_breaks[i]*raw_pix.shape[1]).astype(int)
		for b in range(len(pix_breaks)-1):
			words.append(raw_pix[:, pix_breaks[b]:pix_breaks[b+1] ])

	return words


def preprocess3(img):
	"""an alternate way to chop down the sentences"""
	lines = crop_sentences(img)
	words = []
	window = 7

	for i in range(10):

		line_no = int(np.random.choice([0,1],1)[0])
		raw_pix = np.array(lines[line_no])
		line_end = [20,14][line_no]
		start_char = np.random.choice(range(line_end-window))
		end_char = start_char + window
		start_index, end_index = int(raw_pix.shape[1]*start_char/line_end),int(raw_pix.shape[1]*end_char/line_end)
		words.append(raw_pix[:, start_index:end_index])

	return words



def crop_sentences(img):
	"""cut down on the size of the images by making smaller sentences size chunks"""
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



def crop(img,size):
	"""add necessary white space and crop image"""
	raw_pix = np.array(img)
	crop_img = np.ones((size,size))*255
	
	bottom = np.min((size,raw_pix.shape[0]))
	right = np.min((size,raw_pix.shape[1]))
	
	#center
	top = int((size - bottom)/2)
	left =  int((size - right)/2)

	crop_img[top:top+bottom,left:left+right] = raw_pix[:bottom,:right]
	final_img = Image.fromarray(crop_img)
	return final_img


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

			#check to see if there are more than one style per family
			other_fonts = schema[schema['Family ID'] == family]
			other_fonts  = other_fonts.groupby('Style ID').first()
			other_fonts = len(other_fonts)  #if there are 2 other families

			
			try:
				img = Image.open(file)
				number = ("%03d"%style)[:3]
				processed_imgs = preprocess3(img)

				if (other_fonts - 2 <= 0):
					#drop if its a family with 1 fonts
					processed_imgs = processed_imgs[:6]


				for p_ind in range( len(processed_imgs)):

					#decide wether training or test data
					write_dir = np.random.choice(a=[args.train_dir, args.test_dir], p=[args.percent,1-args.percent])
					fam_path = os.path.join(write_dir , 'fam' + str(family))

					if not os.path.exists(fam_path):
						os.mkdir(fam_path)

					img_name = 'fam' + str(family) + '_' + (number + str(p_ind))[:4]

					#resize/crop and save
					final_img = crop(processed_imgs[p_ind],100)
					if final_img.mode != 'RGB':
						final_img = final_img.convert('RGB')
					final_img.save(fam_path + '/' + img_name + '.png','png')

			except Exception as e:
				print(file, e)



if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))