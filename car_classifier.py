class CarClassifier(object):
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

    def get_features(self):
        """ Extract feature vector from images
        """
        pass
