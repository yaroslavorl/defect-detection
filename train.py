from model import Net
from dataset import DataLoad

import tensorflow as tf
import matplotlib.pyplot as plt

loader = DataLoad('data/', (128, 128), num_frame=16)

train = loader.get_video_data(mode='train', stride=2)
valid = loader.get_video_data(mode='valid', stride=2)
test = loader.get_video_data(mode='test', stride=2)

model = Net()
opt = tf.keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-5)


model.compile(loss='mse', optimizer=opt)
history = model.fit(train, train,
                    batch_size=10,
                    epochs=15,
                    validation_data=(valid, valid),
                    shuffle=False)

print(model.evaluate(test, test, batch_size=10))
model.save('model_weight/model')


plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='vall loss')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.show()
