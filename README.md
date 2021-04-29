# Creating Embeddings and Recognizing Fonts using TensorFlow

This repository holds the codes to train the neural network and produce the embeddings. The embeddings can be used to recognize fonts. The embeddings will be used in https://github.com/ericschulman/fonts_causal_analysis for causal economic analyses. 

## File structure/files
* This repository should have external folders with the data. Here is an example of folder structure under a name `fonts_project`.

```
fonts_project    
└───datasets
│   └─── raw_pangrams
│   └─── main_dataset
│   │   │ Style Sku Family.csv
└───models
└───logs
└───fontnet
```

* We run the code in this repository using Anacondas with Python 3.7 on Ubuntu 18.03. For TensorFlow, version 1.7 or better is required. Install TensorFlow via `conda install tensorflow`.

## Preprocessing 

Run `preprocessing.sh`. This should create the necessary cropped data from original pangram bmp images. 

## Training

Run `train.sh`. We trained until the loss function is between .6-.8. Results may vary. It took us about 36 hours on relatively weak hardware, i.e., I5-6260U CPU @ 1.80GHz × 4 and 16 GB RAM. 

## Cross-validation

First run `gen_pairs.sh`. This should create the necessary data for cross-validation. The `pairs.txt` files will appear in the folder with the test data. There are 2 sets:
* Easy, this is generated by specifying `--diff_style 0`.
* Hard, this is generated by specifying `--diff_style 1`. We test whether the fontnet is trained to recognize font families and not just styles.  

Then run `validate.sh`. This should display statistics about the trained model. You will need to specify the model and log directories. The relevant folders are generated by training a model.


## Saving the embeddings

Run `write_embeddings.sh`.  You will need to specify the model and log directories. The relevant folders are generated by training a model. The result of this script will appear in the `main_dataset` folder. Without modifying the code, the file will be called `embeddings_full.csv`.

## References
* [Han, S., Schulman, E., Grauman, K., & Ramakrishnan, S. (2020). Shapes as product differentiation: Neural network embedding in the analysis of markets for fonts.](https://sites.google.com/site/universs01/mypdf/font_embedding.pdf)

* [Schroff, F., Kalenichenko, D., & Philbin, J. (2015). Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 815-823).](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)

## License

The codes and the dataset (separated shared) for this repository are protected by the Creative Commons non-commerical no-derivative license.
