from sklearn.neural_network import MLPClassifier
from numpy import uint8, frombuffer

from sklearn.metrics import accuracy_score


def load_images(filename):
    with open(filename, "rb") as f:
        data = f.read()

    magic_number = int.from_bytes(data[0 : 4])
    if magic_number != 2051:
        raise ValueError(f"Wrong magic number {magic_number}")

    num = int.from_bytes(data[4: 8])
    rows = int.from_bytes(data[8: 12])
    cols = int.from_bytes(data[12: 16])

    pixels = frombuffer(data, uint8, offset=16)
    images = pixels.reshape(num, rows * cols)

    return images


def load_labels(filename):
    with open(filename, "rb") as f:
        data = f.read()

    magic_number = int.from_bytes(data[0 : 4])
    if magic_number != 2049:
        raise ValueError(f"Wrong magic number {magic_number}")

    num = int.from_bytes(data[4 : 8])

    return frombuffer(data, uint8, offset=8)


x_test = load_images("data/t10k-images.idx3-ubyte")
x_train = load_images("data/train-images.idx3-ubyte")
y_test = load_labels("data/t10k-labels.idx1-ubyte")
y_train = load_labels("data/train-labels.idx1-ubyte")


perceptron = MLPClassifier(
    random_state = 999,
    activation = "relu", # identity, logistic, tanh, relu
    hidden_layer_sizes = (100, 100, 100, 100, 100, 100)
)
perceptron.fit(x_train, y_train)

y_predicted = perceptron.predict(x_test)

res = accuracy_score(y_test, y_predicted)

print(res)

# 100, identity - 0.9003
# 100, logistic - 0.9577
# 100, tanh -     0.9453
# 100, relu -     0.9607
# 100, 150 identity - 0.9183
# 100, 150 logistic - 0.9687
# 100, 150 tanh     - 0.9545
# 100, 150 relu     - 0.9725
# 150, 150 relu     - 0.9722
# 100, 150, 200 relu      - 0.9722
# 100, 100, 100 relu      - 0.9722
# 100, 100, 100, 100 relu - 0.9759
# 100, 100, 100, 100, 100, 100, relu - 0.9787
