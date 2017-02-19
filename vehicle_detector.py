import numpy as np
import cv2
from scipy.ndimage.measurements import label
from scipy.spatial.distance import euclidean

class VehicleDetector:
    """ Vehicle Detector class
    Attributes:
        windows: bounding boxes of windows
        car_clf: car classifier
    """
    def __init__(self, classifier, is_tracking=False):
        """ Initialize vehicle detector
        Attr:
            classifier: car classifier
        """
        self.frame_history = []
        self.is_tracking = is_tracking
        self.car_clf = classifier
        self.car_clf.fit()

        self.windows = []
        self.windows += self.get_windows(x_start_stop = (0,1280),
                                y_start_stop = (400,500), xy_window = (96,64),
                                xy_overlap = (0.9, 0.9))
        self.windows += self.get_windows(x_start_stop = (0,1280),
                                y_start_stop = (400,500), xy_window = (192,128),
                                xy_overlap = (0.80, 0.80))
        self.windows += self.get_windows(x_start_stop = (0,1280),
                                y_start_stop = (430,550), xy_window = (288,192),
                                xy_overlap = (0.5, 0.5))

    def get_windows(self, x_start_stop, y_start_stop, xy_window, xy_overlap):
        """ Get window bounding boxes
        Attr:
            x_start_stop: tuple of start and stop pixels in X direction
            y_start_stop: tuple of start and stop pixels in Y direction
            xy_window: window size
            xy_overlap: fraction of overlap size in x, y
        Returns:
            bounding boxes of windows
        """
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan/nx_pix_per_step) - 1
        ny_windows = np.int(yspan/ny_pix_per_step) - 1
        # Initialize a list to append window positions to
        window_list = []

        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        """ Draw boxes
        Attr:
            img: image to draw boxes on
            bboxes: bounding boxes of given windows
            color: color object
            thick: thickness of box line
        """
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def get_positive_windows(self, img):
        """ Get windows that have cars in it
        Attr:
            img: image to search within
        Returns:
            list of windows having cars
        """
        positive_windows = []
        counter = 1
        for window in self.windows:
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # cv2.imwrite('output_images/' + str(counter) + '.jpg', test_img)
            counter = counter + 1
            if self.car_clf.predict(test_img) == 1:
                positive_windows.append(window)
        return positive_windows


    def add_heat(self, heatmap, bbox_list):
        """ Add heat according to bounding box list
        Attr:
            heatmap: heat map initiliazed to image size
            bbox_list: bounding boxes
        Returns:
            resulted heat map image
        """
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap

    def apply_threshold(self, heatmap, threshold):
        """ Apply threshold to heat image
        Attr:
            heatmap: calculated heat image
            threshold: threshold value
        """
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def does_history_exist(self, centroid):
        for c in self.frame_history:
            if euclidean(c, centroid) < 10:
                return True
        return False

    def draw_labeled_bboxes(self, img, labels):
        """ Draw bounding boxes according to heat map
        Attr:
            img: image ot draw on
            labels: labels of heat map
        """
        centroids = []
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            centroid_x = (bbox[1][0] + bbox[0][0]) / 2.
            centroid_y = (bbox[1][1] + bbox[0][1]) / 2.
            centroids.append((centroid_x, centroid_y))
            # Draw the box on the image
            if self.is_tracking:
                if self.does_history_exist((centroid_x, centroid_y)):
                    cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
            else:
                cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        self.frame_history = centroids
        # Return the image
        return img

    def process_image(self, img):
        """ Process image to find cars
        Attr:
            img: image to process
        Returns:
            image after drwaing boxes around cars
        """
        positive_windows = self.get_positive_windows(img)
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        # Add heat to each box in box list
        heat = self.add_heat(heat,positive_windows)
        # Apply threshold to help remove false positives
        heat = self.apply_threshold(heat,4)
        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        processed_image = self.draw_labeled_bboxes(np.copy(img), labels)
        # processed_image = self.draw_boxes(img, positive_windows)

        return processed_image
