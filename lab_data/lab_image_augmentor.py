import numpy as np
import cv2
import os
import tqdm
import pandas as pd
from joblib import Parallel, delayed
import os
import tqdm
import re
from src.utils import args

class LabImageAugmentations:

    def __init__(self):
        self.img = None
        self.rotated_images = None
        self.flipped_images = None
        self.cropped_images = None
        self.augmented_images = None

    def get_image(self, img_path):
        self.img = cv2.imread(img_path)
        return self.img

    def rotating(self, num_rotations):
        (h, w) = self.img.shape[:2]
        center = (w / 2, h / 2)
        self.rotated_images = []
        for i in range(num_rotations):
            rotations = list(range(0, 180, num_rotations))
            M = cv2.getRotationMatrix2D(center, rotations[i], 1.0)
            rotated = cv2.warpAffine(self.img, M, (w, h))
            self.rotated_images.append(rotated)
        return self.rotated_images

    def flipping(self, img):
        self.flipped_images = []
        originalImage = img
        flipVertical = cv2.flip(originalImage, 0)
        flipHorizontal = cv2.flip(originalImage, 1)
        flipBoth = cv2.flip(originalImage, -1)
        self.flipped_images.append(flipVertical)
        self.flipped_images.append(flipHorizontal)
        self.flipped_images.append(flipBoth)
        return self.flipped_images

    def do_image_augmentations(self, model_df):
        print('AUGMENTING IMAGES')
        save_path = f'./lab_data/checkpoints/inputs/aug_images/'
        os.makedirs(save_path, exist_ok=True)
        model_df = model_df[model_df['is_valid'] == 0]
        input = [(x, y) for x, y in zip(model_df['fname'], model_df['label'])]
        def worker(input):
            count = 0
            image_path, label = input[0], input[1]
            image = re.findall(r'.*\/(.*).png', image_path)[0]
            img = self.get_image(image_path)
            rot = self.rotating(6)
            for im in rot:
                flip = self.flipping(im)
                for flipped in flip:
                    file_name = save_path + f'aug{count}_{image}_{label}.png'
                    cv2.imwrite(file_name, flipped)
                    count += 1
        Parallel(n_jobs=os.cpu_count())(delayed(worker)(i) for i in tqdm.tqdm(input, ncols=80))

