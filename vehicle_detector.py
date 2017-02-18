class VehicleDetector:
    """ Vehicle Detector class
    Attributes:
        windows: bounding boxes of windows
    """
    def __init__(self):
        """ Initialize vehicle detector
        """
        self.windows = []
        self.windows += get_windows(x_start_stop = (0,0),
                                y_start_stop = (400,500), xy_window = (96,96),
                                xy_overlap = (0.75, 0.75))
        self.windows += get_windows(x_start_stop = (0,0),
                                y_start_stop = (400,500), xy_window = (144,144),
                                xy_overlap = (0.75, 0.75))
        self.windows += get_windows(x_start_stop = (0,0),
                                y_start_stop = (430,580), xy_window = (192,192),
                                xy_overlap = (0.75, 0.75))

    def get_windows(x_start_stop, y_start_stop, xy_window, xy_overlap):
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

    def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
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


    def process_image(self):
        pass
