# python 2.7 will make a python 3.x version soon
import sys

import tensorflow as tf

# change this as you see fit
image_path = sys.argv[1]

# Read in the image_data
image_data = tf.gfile.GFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("K:/Tensor/tensorflow-for-poets-2/tf_files/output_labels.txt")]

# Unpersists graph from file
with tf.gfile.GFile("K:/Tensor/tensorflow-for-poets-2/tf_files/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    # Sort to sow labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    print('Predicting Eye Disease')
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.2f)' % (human_string, score*100))
 
f= open("K:/resutlts.txt","w+")

for i in range(1):
       f.write('%s (score in percentage = %.5f)' % (human_string, score))

f.close()

"""node_id = top_k[0] 
human_string= label_lines[node_id]
score = predictions[0][node_id]
print('%s (score = %.5f)' % (human_string, score))"""


