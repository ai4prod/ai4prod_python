# This file help you convert yolov3.weights to yolov3.onnx to be used in Ai4prod inference Library
import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


IMAGE_HEIGHT_ONNX= 416
IMAGE_WIDTH_ONNX= 416
CHANNEL= 3

def detect(save_img=False):
    imgsz = (IMAGE_WIDTH_ONNX, IMAGE_HEIGHT_ONNX) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights = opt.output, opt.source, opt.weights    

    #webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    #if os.path.exists(out):
        #shutil.rmtree(out)  # delete output folder
    
    if(os.path.isdir(out)):
        pass
    else:
        os.makedirs(out)
        print("Output folder do not exists")
    
    #os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    #attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        print(device)
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
        #model=torch.load(weights, map_location=device)['model']
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    #classify = False
    # if classify:
    #     modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
    #     modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        # serve per la quantizzazione unisce BN+Conv2d
        #model.fuse()
        #img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        img = torch.randn(1, CHANNEL, IMAGE_WIDTH_ONNX, IMAGE_HEIGHT_ONNX, device='cpu')
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=True, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return



if __name__ == '__main__':

    # convert Darkent to Ultralytics
    convert("cfg/yolov3-spp.cfg","models/yolov3-spp.weights")


    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='models/yolov3-spp.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='models', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
 
    print(opt)
    
    

    with torch.no_grad():
        detect()
        
        
