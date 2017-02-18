import numpy as np
import cv2
import glob
import pickle
import os

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
        hists = []
        for i in range(3):
            channel_hist = np.histogram(img[:,:,i], bins=32, range=(0,256))
            hists.append(channel_hist[0])

        return np.concatenate((hists[0], hists[1], hists[2]))


    def get_features(self, img_paths):
        """ Extract feature vector from images
        Attr:
            img_paths: list of image paths
        """
        features = []
        for img_path in img_paths:
            feature = []
            img = cv2.imread(img_path)
            color_trans_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
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
                hog_feature = hog(color_trans_img[:,:,channel, orientations=8,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       transform_sqrt=True,
                       visualise=False, feature_vector=True)
                hog_features.append(hog_feature)
            hog_features = np.ravel(hog_features)
            feature.append(hog_features)

            features.append(np.concatenate(feature))

        return features
