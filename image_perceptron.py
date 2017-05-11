import gzip
import pickle
import textwrap

from PIL import Image

from numpy import *

from perceptron import train_n_times, predict

with gzip.open('/home/liam/Downloads/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    pic_arr = reshape(train_set[0][677]*255, (28, 28))
    im = Image.fromarray(pic_arr)
    im.show()

print("678th Digit:\n{}\n".format(train_set[1][677]))
print("678th Digit, 58th pixel:\n{}\n".format(train_set[0][677][57]))


# Train 5

textwrap.fill("TRAINING FIVE", 70)
print("\u2500TRAINING FIVE")

input_vars_5 = (train_set[1] == 5).astype(int)

weights, bias = train_n_times(train_set[0][:40000], input_vars_5[:40000], 0.1, 10)
print("Weights: {}, Bias: {}".format(weights, bias))

count = 0
for input_var, target in zip(train_set[0][40001:50000], input_vars_5[40001:50000]):
    if predict(input_var, weights, bias) == target:
        count += 1

print("Correct {:.2%} of the time".format(count/10000))
