from car_classifier import CarClassifier

if __name__ == "__main__":
    car_img_dir = 'vehicles'
    not_car_img_dir = 'non-vehicles'
    sample_size = 8792
    car_classifier = CarClassifier(car_img_dir=car_img_dir,
        not_car_img_dir=not_car_img_dir,
        sample_size = sample_size)

    car_classifier.fit()
