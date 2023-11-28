import tensorflow as tf
from flask import Flask, request, jsonify
import os
import json
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import warnings
app = Flask(__name__)
tf.get_logger().setLevel('ERROR')          


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



LABEL_FILENAME = 'Clothes_label_map.pbtxt'
PATH_TO_LABELS = 'H:/AI-tfod/Tensorflow/Annotations/Clothes_label_map.pbtxt'
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = 'H:/AI-tfod/Tensorflow/Exported-models/My_Model_final'+ "/saved_model"

print('Loading model...', end='')
start_time = time.time()

detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)   



@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    file.save('H:/AI-tfod/API/image.jpg')
    tf.get_logger().setLevel('ERROR')          

   
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    IMAGE_PATHS =file

    print(IMAGE_PATHS)

    matplotlib.use('Agg')
    warnings.filterwarnings('ignore')   
    def load_image_into_numpy_array(path):
        
        return np.array(Image.open(path))

    print('Running inference for ... ')

    image_np = load_image_into_numpy_array(IMAGE_PATHS)

    input_tensor = tf.convert_to_tensor(image_np)
   
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                       for key, value in detections.items()}
    detections['num_detections'] = num_detections

   
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)


    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False,
            line_thickness=8,
          
    
        )

    Image.fromarray(image_np_with_detections).save('output_image5.jpg')
    print('Done')



    return jsonify({'message': 'File uploaded successfully'})
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5000)