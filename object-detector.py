import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


range_doppler_features = np.load("data/npz_files/umbc_tent_2_cfar.npz", allow_pickle=True)

x_data, y_data = range_doppler_features['out_x'], range_doppler_features['out_y']

classes_values = ['clear', 'not_clear']
classes = len(classes_values)

y_data = tf.keras.utils.to_categorical(y_data - 1, classes)

train_ratio = 0.70
validation_ratio = 0.20
test_ratio = 0.10

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=1 - train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))
x_train = tf.expand_dims(x_train, axis=-1)


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((8, 128, 1), input_shape=x_train.shape[1:]),
    tf.keras.layers.Conv2D(16, (2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(16, (2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(8, (2, 2), activation='relu', padding='same'),  # Use 'same' padding to maintain dimensions
    tf.keras.layers.MaxPooling2D((1, 2)),  # Adjust pooling to only reduce width

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(classes, activation='softmax')
])

# model.summary()

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['acc'])

# this controls the batch size
BATCH_SIZE = 15
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

history = model.fit(train_dataset, epochs=150, validation_data=validation_dataset)

model.save("saved-model/umbc_tent_2_cfar")

predicted_labels = model.predict(x_test)
actual_labels = y_test

label_predicted = np.argmax(predicted_labels, axis=1)
label_actual = np.argmax(actual_labels, axis=1)

results = confusion_matrix(label_actual, label_predicted)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print(f"Training Accuracy: {round(np.average(acc), 3)}")
print(f"Validation Accuracy: {round(np.average(val_acc), 3)}")

epochs = range(1, len(acc) + 1)
fig, axs = plt.subplots(2, 1)

# plot loss
axs[0].plot(epochs, loss, '-', label='Training loss')
axs[0].plot(epochs, val_loss, 'b', label='Validation loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].legend(loc='best')
# plot accuracy
axs[1].plot(epochs, acc, '-', label='Training acc')
axs[1].plot(epochs, val_acc, 'b', label='Validation acc')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].grid(True)
axs[1].legend(loc='best')
plt.show()

ax = plt.subplot()
sns.heatmap(results, annot=True, annot_kws={"size": 20}, ax=ax, fmt='g')

# labels, title and ticks
ax.set_xlabel('Predicted labels', fontsize=12)
ax.set_ylabel('True labels', fontsize=12)
ax.set_title(f'Confusion Matrix for RADAR data with model accuracy {round(np.average(acc), 3)}')
ax.xaxis.set_ticklabels(classes_values, fontsize=15)
ax.yaxis.set_ticklabels(classes_values, fontsize=15)
plt.show()