import os
import random
import argparse
import sys
import numpy as np

class GeneratePairs:
    """
    Generate the pairs.txt file for applying "validate on LFW" on your own datasets.
    """

    def __init__(self, args):
        """
        Parameter data_dir, is your data directory.
        Parameter pairs_filepath, where is the pairs.txt that belongs to.
        Parameter img_ext, is the image data extension for all of your image data.
        """
        self.data_dir = args.data_dir
        self.pairs_filepath = args.saved_dir + 'pairs.txt'
        self.img_ext = '.png'
        self.repeat_times = int(args.repeat_times)
        self.diff_style = int(args.diff_style)
        self.same_word = int(args.same_word)

        #come up with a subset of folders 
        self.folders = []
        for folder in os.listdir(self.data_dir):
            if os.path.isdir(self.data_dir + folder):
                
                files =  os.listdir(self.data_dir + folder)
                num_files = len(files)

                #ensure there are enough examples in the folder, if we need diff styles
                if num_files > 0 and self.diff_style > 0:
                
                    style1 = files[0]
                    for file in files:

                        if style1[-9:-6] != file[-9:-6]:
                            self.folders.append(folder)
                            break

                elif num_files > 0:
                    self.folders.append(folder)

        self.folders = np.random.choice(self.folders, min( int(args.num_classes), len(self.folders)),
         replace=False)



    def generate(self):
        # The repeated times. You can edit this number by yourself
        folder_number = len(self.folders)

        # This step will generate the hearder for pair_list.txt, which contains
        # the number of classes and the repeate times of generate the pair
        with open(self.pairs_filepath,"w") as f:
            f.write(str(self.repeat_times) + "\t" + str(folder_number) + "\n")

        for i in range(self.repeat_times):
            self._generate_matches_pairs()
            self._generate_mismatches_pairs()



    def _generate_matches_pairs(self):
        """
        Generate all matches pairs
        """
        for name in self.folders:

            a = []
            for file in os.listdir(self.data_dir + name):
                if file == ".DS_Store":
                    continue
                a.append(file)

            with open(self.pairs_filepath, "a") as f:
                temp = random.choice(a).split("_") # This line may vary depending on how your images are named.
                w = temp[0]
                l = random.choice(a).split("_")[1].rstrip(self.img_ext)
                r = random.choice(a).split("_")[1].rstrip(self.img_ext)
                
                #enforce different styles
                if self.diff_style > 0:
                    #choose a pair with a different style
                    sub_a = []
                    for file in a:
                        ext = file.split("_")[1].rstrip(self.img_ext)[0:3]
                        if l[0:3] != ext[0:3]:
                            sub_a.append(file)
                    if len(sub_a) > 0:
                        r = random.choice(sub_a).split("_")[1].rstrip(self.img_ext)
                    

                f.write(w + "\t" + l + "\t" + r + "\n")


    def _generate_mismatches_pairs(self):
        """
        Generate all mismatches pairs
        """
        for i, name in enumerate(self.folders):
            if name == ".DS_Store" or name[-3:] == 'txt':
                continue

            remaining = list(self.folders[:])

            del remaining[i]
            remaining_remove_txt = remaining[:]
            for item in remaining:
                if item[-3:] == 'txt':
                    remaining_remove_txt.remove(item)

            remaining = remaining_remove_txt
            other_dir = random.choice(remaining)

            file1 = random.choice(os.listdir(self.data_dir + name))

            #check to see if it is possible to do same word mismatch
            if self.same_word:
                candidate_folders = []
                for folder in remaining:
                    for file in os.listdir(self.data_dir + folder):
                        if file1[-5] == file[-5]:
                            candidate_folders.append(folder)
                            break

                if len(candidate_folders) > 0:
                    remaining = candidate_folders
                    other_dir = random.choice(remaining)

            with open(self.pairs_filepath, "a") as f:
                    
                    file2 = random.choice(os.listdir(self.data_dir + other_dir))
                    if self.same_word:
                        #ensure that false negatives include the same word
                        same_words = []
                        for file in os.listdir(self.data_dir + other_dir):
                            if file1[-5] == file[-5]:
                                same_words.append(file)
                        if len(same_words) > 0:
                            file2 = random.choice(same_words)

                    
                    f.write(name + "\t" + file1.split("_")[1].lstrip("0").rstrip(self.img_ext) \
                     + "\t" + other_dir + "\t" + file2.split("_")[1].lstrip("0").rstrip(self.img_ext) + "\n")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Directory with aligned images.', default='../datasets/small_test/')
    parser.add_argument('--saved_dir', type=str, help='Directory to save pairs.', default='../datasets/small_test/')
    parser.add_argument('--repeat_times', type=str, help='Repeat times to generate pairs', default=3)
    parser.add_argument('--num_classes', type=str, help='number of classes', default=6)
    parser.add_argument('--diff_style', type=str, help='Only include pairs with different styles',default=0)
    parser.add_argument('--same_word', type=str, help='Only include pairs with same word',default=0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    generatePairs = GeneratePairs(parse_arguments(sys.argv[1:]))
generatePairs.generate()