import os
import csv
import shutil
import matplotlib.pyplot as plt

"""
Class for storing configurations for a training job such as
    - whether to use CUDA (if available)
    - batch size to use for training
    - logging interval when training
    - number of epochs to train for
"""


class TrainingConfig:
    def __init__(self, cuda=True, batch_size=128, log_interval=10, epochs=10, output_dir_name='results'):
        self.cuda = cuda
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.epochs = epochs
        self.output_dir_name = output_dir_name


"""
Class for storing configurations for a dataset used in training
    - size of each image in dataset (width/height) since
        - Example: 96 for STL10, 28 for MNIST
    - number of channels for images in dataset
        - Example: 3 for RGB, 1 for grayscale
"""


class DatasetConfig:
    def __init__(self, image_size, channels):
        self.image_size = image_size
        self.channels = channels


"""
Function to write a list of data to a .csv file
"""


def write_to_csv(data=None, file_name=None):
    assert data is not None and file_name is not None

    with open(file_name, 'w', newline='') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(data)


"""
Function to create a directory, if one already exists, remove it and create an empty one
"""


def make_directory(dir_name='results'):
    output = dir_name
    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output)


"""
Function to convert a .csv file (model training / test loss) into a 
matplotlib .png plot
"""


def plot_from_csv(csv_file, output_file):
    data = []
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            for num in row:
                data.append(float(num))
    plt.plot(data)
    plt.savefig(f"{output_file}.png")
