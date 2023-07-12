from dataset import DataLoad
import matplotlib.pyplot as plt
import numpy as np
import keras

model = keras.models.load_model('model_v1')

num_frames = 150
test = DataLoad('adnormal', (128, 128), 16).get_video_data(mode='test')

rec = model.predict(test, batch_size=10)
reconstructed_sequences = rec.reshape(-1, 128, 128, 1)
test = test.reshape(-1, 128, 128, 1)
cost = np.array([np.linalg.norm((test[i] - rec[i]) for i in range(0, num_frames))])

s = (cost - np.min(cost)) / np.max(cost)
sr = 1.0 - s

plt.plot(sr)
plt.ylabel('regularity score')
plt.xlabel('frame')
plt.show()