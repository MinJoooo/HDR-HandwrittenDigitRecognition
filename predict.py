import numpy as np
import argparse
import cv2
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neural_network import CNN
from PIL import Image, ImageFilter


# Parse the Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save_model", type=int, default=-1)
ap.add_argument("-l", "--load_model", type=int, default=-1)
ap.add_argument("-w", "--save_weights", type=str)
ap.add_argument("-m", type=int, default=-1)
args = vars(ap.parse_args())

if args["m"] == 1:
    args["load_model"] = 1
    args["save_weights"] = "cnn_weights.hdf5"


def Predict():
    # Read/Download MNIST Dataset
    print('Loading MNIST Dataset...')
    # dataset = fetch_mldata('MNIST Original')
    dataset = fetch_openml('mnist_784')

    # Read the MNIST data as array of 784 pixels and convert to 28x28 image matrix.
    mnist_data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
    mnist_data = mnist_data[:, np.newaxis, :, :]

    # Divide data into testing and training sets.
    train_img, test_img, train_labels, test_labels = train_test_split(mnist_data/255.0, dataset.target.astype("int"), test_size=0.1)

    # Now each image rows and columns are of 28x28 matrix type.
    img_rows, img_columns = 28, 28

    # Transform training and testing data to 10 classes in range [0,classes]; num of classes = 0 to 9 = 10 classes
    total_classes = 10 # 0 to 9 labels
    train_labels = np_utils.to_categorical(train_labels, 10)
    test_labels = np_utils.to_categorical(test_labels, 10)

    # Define and compile the SGD optimizer and CNN model.
    print('Compiling model...')
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    clf = CNN.build(width=28, height=28, depth=1, total_classes=10, Saved_Weights_Path=args["save_weights"] if args["load_model"] > 0 else None)
    clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    # Initially train and test the model; If weight saved already, load the weights using arguments.
    b_size = 128  # batch size
    num_epoch = 20  # number of epochs
    verb = 1  # verbose

    # If weights saved and argument load_model; Load the pre-trained model.
    if args["load_model"] < 0:
        print('Training the Model...')
        clf.fit(train_img, train_labels, batch_size=b_size, epochs=num_epoch, verbose=verb)

        # Evaluate accuracy and loss function of test data.
        print('Evaluating Accuracy and Loss Function...')
        loss, accuracy = clf.evaluate(test_img, test_labels, batch_size=128, verbose=1)
        print('Accuracy of Model: {:.2f}%'.format(accuracy * 100))

    # Save the pre-trained model.
    if args["save_model"] > 0:
        print('Saving weights to file...')
        clf.save_weights(args["save_weights"], overwrite=True)

    # Read test data
    im = Image.open("draw.png")
    np.set_printoptions(threshold=np.inf) # show all elements of array
    resized_image = im.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    pix = np.array(resized_image)
    pix2 = cv2.cvtColor(pix, cv2.COLOR_BGR2GRAY) # convert colors to black and white
    im_data = pix2.reshape((-1, 1, 28, 28))

    # # Identify im_data as array of 0 and 1.
    # im_data2 = pix2.reshape((1, 1, 28, 28))
    # im_data3 = np.zeros((1, 1, 28, 28))
    # for i in range(0, 28):
    #     for j in range(0, 28):
    #         if(im_data2[0][0][i][j] > 0):
    #             im_data3[0][0][i][j] = 1
    # im_data3 = np.array(im_data3, np.int32)  # convert elements from decimal to integer
    # print(im_data3)s

    # Predict test data
    var = clf.predict(im_data)
    pred = var.argmax(axis=1)
    print("Here is the result:", pred, "\n")
    image = (im_data[0][0] * 255).astype("uint8") # output by changing image size
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, str(pred), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow('Digits', image) # image output after switching to digit data
    cv2.waitKey(0)