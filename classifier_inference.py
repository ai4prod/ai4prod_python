import onnxruntime
import onnx
from timeit import Timer
import numpy as np
from utils import *
import cv2
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    model = 'resnet50finetuned.onnx'
    root = './images/'
    # Check device
    print(onnxruntime.get_device())
    session = onnxruntime.InferenceSession(model, sess_options= None)
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name

    with open('ImagenetClasses.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    # Open Images
    for filename in os.listdir(root):

        image = cv2.imread(os.path.join(root, filename))
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        image_data = np.array(image).transpose(2, 0, 1)
        input_data = preprocess_onnx(image_data)
        raw_result = session.run([], {input_name: input_data})
        #uncomment to check speed with timeit
        #speed('session.run([], {input_name: input_data})')
        res = postprocess(raw_result)
        
        np.savetxt("results.txt", np.array(input_data.flatten()), delimiter = ' ')
        idx = np.argmax(res)

        print('========================================')
        print('Final top prediction is: ' + classes[idx])
        print('========================================\n')

        sort_idx = np.flip(np.squeeze(np.argsort(res)))
        print('============ Percentage: ============================')
        print(res[idx])
        print('===========================================================')
