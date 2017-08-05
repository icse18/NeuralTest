from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

def encode_label(label):
    return int(label)

def read_label_file(file):
    f = open(file, 'r')
    filepaths = []
    labels = []
    for line in f:
        filepath, label = line.split(",")
        filepaths.append(filepath)
        labels.append(label)
    return filepaths, labels

# dataset_path      = "/path/to/out/dataset/mnist/"
# test_labels_file  = "test-labels.csv"
# train_labels_file = "train-labels.csv"


# reading labels and file path
#train_filepaths, train_labels = read_label_file(dataset_path + train_labels_file)
#test_filepaths, test_labels = read_label_file(dataset_path + test_labels_file)

all_vectors = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)

# Let's partition the data
partitions = [0] * len(all_filepaths)
partitions[:test_set_size] = [1] * test_set_size
random.shuffle(partitions)
