from car_classifier import CarClassifier
from vehicle_detector import VehicleDetector
import glob
import cv2
import numpy as np
import ntpath
import argparse
from moviepy.editor import VideoFileClip

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
    if type == 'v':
        vehicle_detector = VehicleDetector(classifier=car_classifier, is_tracking = True)
        clip = VideoFileClip("./project_video.mp4")
        # cut = clip.subclip(0.04,0.1)
        output_video = "./output_video/project_video.mp4"
        output_clip = clip.fl_image(vehicle_detector.process_image)
        output_clip.write_videofile(output_video, audio=False)
    elif type == 'i':
        vehicle_detector = VehicleDetector(classifier=car_classifier)
        images = glob.glob('test_images/*.jpg')
        for idx, fname in enumerate(images):
            print('Processing image ', idx)
            image = cv2.imread(fname)
            processed_image = vehicle_detector.process_image(image)
            print('Processing done!!! ', idx)
            output_filename = 'output_images/' + ntpath.basename(fname)
            cv2.imwrite(output_filename, processed_image)
    else:
        print('Invalid type requested')

def extract_video():
    vidcap = cv2.VideoCapture('./project_video.mp4')
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      cv2.imwrite("v_images/frame%d.jpg" % count, image)
      count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='type i/v/d')
    args = parser.parse_args()

    if args.type == 'd':
        doc()
    else:
        detect_vehicles(args.type)
