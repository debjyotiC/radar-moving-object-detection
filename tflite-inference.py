import numpy as np
import tensorflow as tf
import time

type_of_quantization = "default"
model_path = f"saved-tflite-model/range-doppler-tent-2-{type_of_quantization}.tflite"

range_doppler_features = np.load("data/npz_files/umbc_tent_cfar.npz", allow_pickle=True)
x_data, y_data = range_doppler_features['out_x'], range_doppler_features['out_y']

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_index = input_details["index"]
classes_values = ['clear', 'not_clear']

print(f"Model type {type_of_quantization}")

for i, true_label in enumerate(y_data):
    data = x_data[i]

    in_tensor = np.float32(data.reshape(1, data.shape[0], data.shape[1], 1))
    # print(in_tensor.shape)
    start_time = time.time()
    interpreter.set_tensor(input_index, in_tensor)
    interpreter.invoke()
    classes = interpreter.get_tensor(output_details['index'])
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000.0
    confidence_scores = np.squeeze(classes)
    max_index = np.argmax(confidence_scores)
    max_value = confidence_scores[max_index]
    print(max_value)
    pred = np.argmax(classes[0])
    print(f"Inference time: {elapsed_time} ms")
    print("Pred. class label ", classes_values[pred], "for true label ", classes_values[true_label-1])
