from numpy import frombuffer, uint8


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


if __name__ == "__main__":
    result = load_labels("data/train-labels.idx1-ubyte")

    print(result)
