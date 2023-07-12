import os
import cv2
import numpy as np


class DataLoad:

    def __init__(self, data_path, size_img, num_frame):
        self.data_path = data_path
        self.size_img = size_img
        self.num_frame = num_frame

    def augmentation(self, frames, num_frame, stride):

        video_data = []
        video = np.zeros((num_frame, self.size_img[0], self.size_img[1], 1))

        count = 0
        for start in range(0, stride):
            for i in range(start, len(frames), stride):
                video[count, :, :, 0] = frames[i]
                count += 1
                if count == num_frame:
                    video_data.append(video)
                    count = 0
        return video_data

    def get_video_data(self, mode='train', stride=1):

        video_data = []
        path = os.path.join(self.data_path, mode)

        for video in os.listdir(path):
            dir_path = os.path.join(path, video)
            if os.path.isdir(dir_path):
                frames = []
                for img in sorted(os.listdir(dir_path), key=self.sorted_key):

                    img_path = os.path.join(dir_path, img)
                    img = cv2.imread(img_path, 0)
                    img = cv2.resize(img, self.size_img)
                    img = np.array(img, dtype=np.float32) / 255
                    frames.append(img)

                for stride in range(1, stride + 1):
                    video_data.extend(self.augmentation(frames, self.num_frame, stride))
        return video_data

    @staticmethod
    def sorted_key(x):
        return int(x.split('.')[0])

