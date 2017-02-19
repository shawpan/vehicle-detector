#Vehicle Detection and Tracking

## Demo

<a href="http://www.youtube.com/watch?feature=player_embedded&v=Hswv-lF-zj8
" target="_blank"><img src="http://img.youtube.com/vi/Hswv-lF-zj8/0.jpg"
alt="Track 1" width="608" border="10" /></a>

---


[//]: # (Image References)

[car_not_car]: ./doc/car_not_car.png
[car]: ./doc/car.png
[hog]: ./doc/hog.jpg
[video1]: ./output_video/project_video.mp4

##Histogram of Oriented Gradients (HOG)

The code for this step is contained in the `get_feature()` method of `CarClassifier` class in `car_classifier.py` file.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Car and Nor Car][car_not_car]

below is the code for extracting combined features of an image

```python
def get_feature(self, img):
        """ Get feature of img
        Attr:
            img: image object
        Returns:
            feature vector
        """
        feature = []
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

        return np.concatenate(feature)
```
**Car and Corresponding HOG feature visualization**

**Car**

![car][car]

**HOG Feature**

![hog][hog]

### HOG parameters.

I tried various combinations of parameters and looked for experiment results in various blog posts. Finally the following parameters worked well

```
orientations=8
pixels_per_cell=(8, 8)
cells_per_block=(2, 2)
transform_sqrt=True
```

#### Training car classifier

I trained a linear SVM in `CarClassifier` class in `car_classifier.py` using the combined features extracted from image. The training is done in `fit()` method of `CarClassifier` class

```python
def fit(self):
        """ Fit classifier to car and not car data
        """
        if os.path.isfile('model.p'):
            with open('model.p', 'rb') as data_file:
                data = pickle.load(data_file)
                self.model = data['model']
                self.scaler = data['scaler']
            return self.model

        x_train, y_train, x_test, y_test = self.get_data()
        svc = LinearSVC(max_iter=20000)
        svc.fit(x_train, y_train)
        data = {
            'model' : svc,
            'scaler' : self.scaler
        }
        with open('model.p', "wb") as data_file:
            pickle.dump(data, data_file)
        self.model = svc
        return self.model
```

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

