import numpy as np
from timeit import Timer

def speed(inst, number=20, repeat=20):
    timer = Timer(inst, globals=globals())
    raw = np.array(timer.repeat(repeat, number=number))
    ave = raw.sum() / len(raw) / number
    mi, ma = raw.min() / number, raw.max() / number
    print("Average %1.3g min=%1.3g max=%1.3g" % (ave, mi, ma))
    return ave

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(res):
    return softmax(np.array(res)).tolist()

def preprocess_onnx(input_data):
    """
    ONNX/Pytorch compliant conversion of images into Tensor and Normalization 
    of them based on the mean and std of ImageNet dataset

    """
    # Convert input data into tensors
    img_data = input_data.astype('float32')
    img_data = img_data.reshape(1, 3, 224, 224)
    # Normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    std_vec = np.array([0.229, 0.224, 0.225])
    norm_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_data[:,i,:,:] = (img_data[:,i,:,:]/255 - mean_vec[i])/std_vec[i]
    return norm_data