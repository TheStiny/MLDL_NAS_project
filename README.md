# projectMLDL
Source code to run our block-based evolution NAS approach for Tiny Visual Wake Words, relative to Project4a - Machine Learning and Deep Learning Final Course Project

download_resize_dataset.py contains the code to download and process the images

genetic.py contains the genetic algorithm

blocksreconstruction.py allows to stack blocks and build neural network architectures

metrics.py allows to compute the metrics and the number of parameters and Flops of a neural network

main.py contains the code to run the search algorithm, to find the best model with the highest score, while respecting hardware constraints, and to train the final model


