# This script will perform object detection using "mynet"
# Images are specified by user input in the console
# Expects that py-faster-rcnn is set up

# Setup some python paths
import sys, os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, os.path.join(os.environ['FRCN_ROOT'], path))

# Add caffe to PYTHONPATH
add_path('caffe-fast-rcnn/python')

# Add lib to PYTHONPATH
add_path('lib')

# Include fast_rccn and misc libraries
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

#
# Parse input arguments
#
def parse_args():
    parser = argparse.ArgumentParser(description='Detects objects in a image using a net trained on the udacity dataset')
    parser.add_argument('--gpu_id', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--use_cpu', dest='use_cpu',
                        help='Set to 1 to use the CPU [0]',
                        default=0, type=int)
    parser.add_argument('--images', dest='images_file',
                        help='Text file containing the image to process. If not provided, the user can specify images in the console',
                        default=None, type=str)
    parser.add_argument('--output', dest='output_dir',
                        help='Results are saved to this directory. If not provided, plots are shown on-screen',
                        default=None, type=str)
    parser.add_argument('--output_suffix', dest='output_suffix',
                        help='File suffix appended to the output',
                        default="_detect", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    # Create a list of classes and a map of trained nets
    CLASSES = ('__background__', 'car', 'truck', 'pedestrian', 'trafficLight', 'biker')

    # Enable RPN (Region Proposal Net) and set cpu or gpu mode
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    if args.use_cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = 0

    # Set models directory
    cfg.MODELS_DIR = os.path.join(os.environ['FRCN_ROOT'], "models")

    # Load the trained net into Caffe
    prototxt = os.path.join(cfg.MODELS_DIR, 'mynet', 'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models', 'mynet_faster_rcnn_final.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(("%s not found") % caffemodel)
    if not os.path.isfile(prototxt):
        raise IOError(("%s not found") % prototxt)

    print('Loading network %s' % caffemodel)    
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print('Network loaded')

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in range(2):
        _, _= im_detect(net, im)

    # If an image list is provided, process it
    image_list = None
    if args.images_file:
    	image_list = []
    	with open(args.images_file) as f:
    		for line in f.readlines():
    			image_list.append(line.strip())

    # If no image list was provided, loop forever and ask for user input for every loop
    # If provided, process all input images
    while not args.images_file or image_list:
        if not image_list:
            # Load input image
            image_file = input("Image path: ")
            if image_file == "quit": break
        else:
            image_file = os.path.join(os.path.dirname(args.images_file), image_list.pop())

        # Check if file exists
        if not os.path.isfile(image_file):
            # Try to load from data/MyData/data/Images
            tmp = os.path.join(cfg.DATA_DIR, 'MyData', 'data', 'Images', image_file)
            if not os.path.isfile(tmp):
                print("Image file %s not found" % image_file)
                continue
            else:
                image_file = tmp

        print("Detecting objects in image %s" % image_file)

        # Load image
        im = cv2.imread(image_file)

        # Perform object detection (im_detect)
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, im)
        timer.toc()
        print(('Detection took {:.3f}s for {:d} object proposals').format(timer.total_time, boxes.shape[0]))

        ### Draw detected bounding boxes ###
        def vis_detections(im, class_name, dets, ax, thresh=0.5):
            inds = np.where(dets[:, -1] >= thresh)[0] # inds = the list of the indices of
            if len(inds) == 0:                        # objects whose score are above tresh
                return
            
            # For every object we detected
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]

                # Draw bounding box
                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor='red', linewidth=3.5)
                    )
                
                # Draw class name and score
                ax.text(bbox[0], bbox[1] - 2,
                        '{:s} {:.3f}'.format(class_name, score),
                        bbox=dict(facecolor='blue', alpha=0.5),
                        fontsize=14, color='white')
                


        # Plot the detection results
        #
        # Convert image from BGR to RGB
        imrgb = im[:, :, (2, 1, 0)]
            
        # Plot image in figure
        fig, ax = plt.subplots(figsize=(12, 12))
        fig.canvas.set_window_title(image_file) 
        ax.imshow(imrgb, aspect='equal')

        # Visualize detections for each class
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        ax.set_title(('detection threshold >= %.1f\ndetection time = %.3f (using %s)') % (CONF_THRESH, timer.total_time, "CPU" if args.use_cpu else "GPU"), fontsize=14)
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)] # Grabs all the (300) rectangles for this class
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]) # Create a list dets[n] = [x0, y0, x1, y1, score]
                             ).astype(np.float32)                   # Basically, append the score to the rectangle coordinates
            keep = nms(dets, NMS_THRESH) # Non-maximum supression
            dets = dets[keep, :] # Remove non-maxima rectangles from dets
            vis_detections(im, cls, dets, ax, thresh=CONF_THRESH) # Visualize detections

        # Draw and show figure
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        if args.output_dir:
        	plt.savefig(os.path.join(args.output_dir, os.path.splitext(os.path.basename(image_file))[0] + args.output_suffix + ".png"))
        else:
        	plt.show()

        # DEBUG: For each class, print it's max score
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            print(cls, scores[:, cls_ind].max())
