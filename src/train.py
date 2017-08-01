import csv
import numpy as np
import imageio
import argparse
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from model import v1, preprocess


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('dir', help='Dataset directory')
args = parser.parse_args()


# Generate batches of data
# Randomly selects a bin
# Randomly samples from bin
# Randomly flips data
# TODO
## Randomly skew
def gen(bins, batch_size=32):
    bin_num = len(bins)
    images = []
    measurements = []
    while 1: # Loop forever so the generator never terminates
        # uniform random on bins
        bin_idx = np.random.randint(0,bin_num)
        bin_samples = bins[bin_idx]

        # Uniform random on samples in bin
        sample_num = len(bin_samples)
        if sample_num == 0: #empty bin
            continue
        sample_idx = np.random.randint(0,sample_num)
        sample = bin_samples[sample_idx]

        # Extract sample
        steering = float(sample["steering"])
        filename = args.dir + "/" + sample["img_file"]
        image = preprocess(imageio.imread(filename))

        #randomly flip sample
        flip = np.random.randint(2)
        if(flip):
            image = np.fliplr(image)
            steering = -steering

        images.append(image)
        measurements.append(steering)

        #return batch
        if(len(images) >= batch_size):
            X = np.array(images)
            y = np.array(measurements)
            images = []
            measurements = []
            yield shuffle(X,y)

def bin_samples(samples,num_bins=21):
    samples =np.array(samples)
    samples = shuffle(samples)

    #Extract steering angles
    angs = []
    for sample in samples:
        angs.append(float(sample["steering"]))

    #Bin data based on steering angle
    bin_lim = np.linspace(-1.1, 1.1, num=num_bins-1,endpoint=True)
    dig = np.digitize(angs,bin_lim,right=False)

    #place data in bins
    bins = []
    max_bin_size = 0
    for i in range(num_bins):
        idx = np.where(dig == i)
        bins.append(samples[idx])
        if len(idx[0]) > max_bin_size:
            max_bin_size = len(idx[0])

    return bins, max_bin_size


if __name__=="__main__":
    # parse csv
    print("Parsing CSV")
    samples = None
    with open(args.dir + '/labels.csv') as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = next(reader)
        samples = [dict(zip(header, map(str, row))) for row in reader]


    # Split samples into training/test set
    print("Splitting data")
    samples = shuffle(samples)
    train_samples, test_samples = train_test_split(samples, test_size=0.1)

    # Bin data based on steering angle
    print("Binning data")
    num_bins = 121
    train_bins,train_max_bin_size = bin_samples(train_samples,num_bins)
    test_bins,test_max_bin_size = bin_samples(test_samples,num_bins)

    # Training parameters
    batch_size = 64
    augmentation_factor = 2 #flip(2)
    epochs = 4
    train_epoch_size = int(len(train_samples) * 60 / batch_size) * batch_size
    test_epoch_size = int(len(test_samples) * 20 / batch_size) * batch_size
    #train_epoch_size = int(num_bins * train_max_bin_size * augmentation_factor / batch_size) * batch_size
    #test_epoch_size = int(num_bins * test_max_bin_size * augmentation_factor / batch_size) * batch_size
    print("Training samples: {}".format(train_epoch_size))
    print("Testing samples: {}".format(test_epoch_size))

    # Create generators
    train_generator = gen(train_bins, batch_size=batch_size)
    validation_generator = gen(test_bins, batch_size=batch_size)

    # fit model
    print("Fitting model")
    model = v1()
    model.fit_generator(train_generator, samples_per_epoch= train_epoch_size,\
                validation_data=validation_generator, \
                nb_val_samples=test_epoch_size, nb_epoch=epochs)

    model.save("model.h5")
