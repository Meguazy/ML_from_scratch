import tensorflow
import numpy as np

from tensorflow import keras

from keras import models, layers, optimizers, datasets, utils, callbacks

def build(input_shape, classes):
    model = models.Sequential()

    # CONV _ RELU _ POOL
    model.add(layers.Convolution2D(20, (5, 5), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # CONV _ RELU _ POOL
    model.add(layers.Convolution2D(50, (5, 5), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten -> RELU layers
    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation="relu"))

    # a softmax classifier
    model.add(layers.Dense(classes, activation="softmax"))

    return model

# define the CNN
def build2(input_shape, classes):
    model = models.Sequential()
    model.add(layers.Convolution2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes, activation="softmax"))

    return model

def build_model(input_shape, classes):
    model = models.Sequential()

    # 1st block
    model.add(layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    # 2nd block
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # 3rd block
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4))

    # dense
    model.add(layers.Flatten())
    model.add(layers.Dense(classes, activation="softmax"))

    return model

if __name__ == "__main__":
    EPOCHS = 5
    BATCH_SIZE = 128
    VERBOSE = 1
    OPTIMIZER = optimizers.Adam()
    VALIDATION_SPLIT = 0.95
    IMG_ROWS, IMG_COLS = 28, 28 # input image dimensions
    INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
    NB_CLASSES = 10 # number of outputs = number of digits

    # data: shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    # reshape
    X_train = X_train.reshape((60000, IMG_ROWS, IMG_COLS, 1))
    X_test = X_test.reshape((10000, IMG_ROWS, IMG_COLS, 1))

    # normalize
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # cast
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    # convert class vectors to binary class matrices
    y_train = utils.to_categorical(y_train, NB_CLASSES)
    y_test = utils.to_categorical(y_test, NB_CLASSES)

    # initialize the optimizer and model
    model = build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
    model.summary()

    # use TensorBoard, princess Aurora!
    callbacksaa = [callbacks.TensorBoard(log_dir="./logs")]

    # fit
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT, callbacks=callbacksaa)

    score = model.evaluate(X_test, y_test, verbose=VERBOSE)

    print("Test score:", score[0])
    print("Test accuracy:", score[1])

    print("Done!")

    """ # CIFAR-10 is next!
    print("CIFAR-10 is next!")
    # CIFAR-10 is a set of 60,000 color images (32x32 pixels) with 10 classes
    IMG_CHANNELS = 3
    IMG_ROWS, IMG_COLS = 32, 32

    # constants
    BATCH_SIZE = 128
    NB_EPOCH = 50
    NB_CLASSES = 10
    VERBOSE = 1
    VALIDATION_SPLIT = 0.2
    OPTIM = optimizers.RMSprop()

    # data: shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

    # normalize
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)

    # convert to categorical
    Y_train = utils.to_categorical(y_train, NB_CLASSES)
    Y_test = utils.to_categorical(y_test, NB_CLASSES)

    print(X_train.shape[0], "train samples")
    # image augmentation
    datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    datagen.fit(X_train)
    print(X_train.shape[0], "train samples")


    # build model
    model = build_model(input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), classes=NB_CLASSES)
    model.summary()

    # use TensorBoard, princess Aurora!
    callbacksbb = [callbacks.TensorBoard(log_dir="./logs2")]

    # train
    model.compile(loss="categorical_crossentropy", optimizer=OPTIM, metrics=["accuracy"])
    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT, verbose=VERBOSE, callbacks=callbacksbb)

    score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)

    print("Test score:", score[0])
    print("Test accuracy:", score[1]) """