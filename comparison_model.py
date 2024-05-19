import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from pathlib import Path
from keras import applications
from keras import layers
from keras import losses
from keras import ops
from keras import optimizers
from keras import metrics
from keras import Model
from keras.api.applications import resnet

target_shape = (200, 200)



