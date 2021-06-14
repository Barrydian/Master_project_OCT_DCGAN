# import all library 
import tensorflow as tf
import numpy as np
import os
import cv2
import random
import scipy.misc
import shutil
random_dim = 100

slim = tf.contrib.slim

HEIGHT, WIDTH, CHANNEL = 128, 128, 3
BATCH_SIZE = 64
EPOCH = 10000
version = 'newOctImages'
newOCT_path = './' + version

# Cgenerator 
nb_couch_gen4, nb_couch_gen8, nb_couch_gen16, nb_couch_gen32, nb_couch_gen64 = 512, 256, 128, 64, 32 # nb_couch_genhannel num
s4 = 4

# Cdiscriminator 
ch_disc2, ch_disc4, ch_disc8, ch_disc16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 512


