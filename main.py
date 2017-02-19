from car_classifier import CarClassifier
from vehicle_detector import VehicleDetector
import glob
import cv2
import numpy as np
import ntpath
import argparse
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

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
    img = cv2.imread('doc/car.png')
    car_classifier.get_feature(img, vis=True)
    img = cv2.imread('doc/car.png')
    vehicle_detector = VehicleDetector(classifier=car_classifier)

    img = cv2.imread('doc/test1.jpg')
    processed_image = vehicle_detector.draw_boxes(img, vehicle_detector.windows)
    cv2.imwrite('doc/sliding_windows.jpg', processed_image)

    positive_windows = vehicle_detector.get_positive_windows(img)
    processed_image = vehicle_detector.draw_boxes(img, positive_windows)
    cv2.imwrite('doc/sliding_window_positives.jpg', processed_image)

    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = vehicle_detector.add_heat(heat,positive_windows)
    # Apply threshold to help remove false positives
    heat = vehicle_detector.apply_threshold(heat,4)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    cv2.imwrite('doc/heat.jpg', heat * 255)

    labels = label(heatmap)
    processed_image = vehicle_detector.draw_labeled_bboxes(np.copy(img), labels)
    cv2.imwrite('doc/result.jpg', processed_image)

def detect_vehicles(type):
    car_classifier = CarClassifier(car_img_dir=car_img_dir,
        not_car_img_dir=not_car_img_dir,
        sample_size = sample_size)
    if type == 'v':
        vehicle_detector = VehicleDetector(classifier=car_classifier, is_tracking = True)
        clip = VideoFileClip("./project_video.mp4")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='type i/v/d')
    args = parser.parse_args()

    if args.type == 'd':
        doc()
    else:
        detect_vehicles(args.type)
