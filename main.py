from car_classifier import CarClassifier
from vehicle_detector import VehicleDetector
import glob
import cv2
import numpy as np
import ntpath

car_img_dir = 'vehicles'
not_car_img_dir = 'non-vehicles'
sample_size = 8792

def doc():
    """ Run pipeline to generate documentation images
    """
    car_classifier = CarClassifier(car_img_dir=car_img_dir,
        not_car_img_dir=not_car_img_dir,
        sample_size = sample_size)
    car_classifier.fit()
    car_classifier.describe()
    print('Car predicted as : ',  car_classifier.predict('doc/car.png'))
    print('NotCar predicted as : ',  car_classifier.predict('doc/notcar.png'))

def detect_vehicles(type):
    car_classifier = CarClassifier(car_img_dir=car_img_dir,
        not_car_img_dir=not_car_img_dir,
        sample_size = sample_size)
    vehicle_detector = VehicleDetector(classifier=car_classifier)
    if type == 'v':
        pass
    elif type == 'i':
        images = glob.glob('test_images/test*.jpg')
        for idx, fname in enumerate(images):
            print('Processing image ', idx)
            image = cv2.imread(fname)
            processed_image = vehicle_detector.process_image(image)
            print('Processing done!!! ', idx)
            output_filename = 'output_images/' + ntpath.basename(fname)
            cv2.imwrite(output_filename, processed_image)
    else:
        print('Invalid type requested')

if __name__ == "__main__":
    detect_vehicles('i')
