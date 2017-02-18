import numpy as np
import cv2
import glob
import pickle
import os
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

class CarClassifier:
    """ Classifier for car object
    Attributes:
        car_img_dir: path to car images
        not_car_img_dir: path to not car images
        sample_size: number of images to be used to train classifier
    """

    def __init__(self, car_img_dir, not_car_img_dir, sample_size):
        """ Initialize class members
        Attr:
            car_img_dir: path to car images
            not_car_img_dir: path to not car images
            sample_size: number of images to be used to train classifier
        """
        self.car_img_dir = car_img_dir
        self.not_car_img_dir = not_car_img_dir
        self.sample_size = sample_size

    def get_color_hist(self, img):
        """ Get color histogram
        Attr:
            img: image object
        Returns:
            histogram array
        """
        hists = []
        for i in range(3):
            channel_hist = np.histogram(img[:,:,i], bins=32, range=(0,256))
            hists.append(channel_hist[0])

        return np.concatenate((hists[0], hists[1], hists[2]))

    def get_features(self, img_paths):
        """ Extract feature vector from images
        Attr:
            img_paths: list of image paths
        Returns:
            features vector
        """
        features = []
        for img_path in img_paths:
            feature = []
            img = cv2.imread(img_path)
            color_trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            img_shape = color_trans_img.shape

            # Spatial feature
            spatial_features = cv2.resize(color_trans_img, (16,16)).ravel()
            feature.append(spatial_features)
            # Histogram feature
            hist_features = self.get_color_hist(color_trans_img)
            feature.append(hist_features)
            # Hog feature
            hog_features = []
            for channel in range(img_shape[2]):
                hog_feature = hog(color_trans_img[:,:,channel], orientations=8,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       transform_sqrt=True,
                       visualise=False, feature_vector=True)
                hog_features.append(hog_feature)
            hog_features = np.ravel(hog_features)
            feature.append(hog_features)

            features.append(np.concatenate(feature))

        return features

    def get_data(self):
        """ Prepare train and test data set from car and not car images
        Returns:
            (x_train, y_train, x_test, y_test)
        """
        print("Preparing train and test data...")
        # If data exists then return
        if os.path.isfile('data.p'):
            with open('data.p', 'rb') as data_file:
                data = pickle.load(data_file)
            print("Done preparing train and test data")
            return (data['x_train'], data['y_train'], data['x_test'], data['y_test'])

        car_images = glob.glob(self.car_img_dir + '/**/*.png', recursive=True)
        car_images = car_images[:self.sample_size]

        not_car_images = glob.glob(self.not_car_img_dir + '/**/*.png', recursive=True)
        not_car_images = not_car_images[:self.sample_size]

        car_features = self.get_features(car_images)
        not_car_features = self.get_features(not_car_images)

        features = np.vstack((car_features, not_car_features)).astype(np.float64)
        scaler = StandardScaler().fit(features)
        scaled_features = scaler.transform(features)
        labels = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))
        x_train, x_test, y_train, y_test = train_test_split(
            scaled_features, labels, test_size=0.2, random_state=np.random.randint(0, 100))

        data = {
            'x_train' : x_train,
            'y_train' : y_train,
            'x_test'  : x_test,
            'y_test'  : y_test
        }
        # Save as a file
        with open('data.p', "wb") as data_file:
            pickle.dump(data, data_file)
        print("Done preparing train and test data")

        return (x_train, y_train, x_test, y_test)

    def fit(self):
        """ Fit classifier to car and not car data
        """
        x_train, y_train, x_test, y_test = self.get_data()
        print('Training set : ', x_train.shape)
        print('Test set : ', x_test.shape)
        print("Training started...")
        svc = LinearSVC(max_iter=20000)
        svc.fit(x_train, y_train)
        print("Finished training.")
