import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(output_graph.pb)
tflite_model = converter.convert()
open("eye.tflite", "wb").write(tflite_model)
