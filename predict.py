import numpy as np
import tensorflow as tf
from Utills import output_groundtruth
from PIL import Image
import cv2
from model import build_model
from data_preprocessing import BatchGenerator
#import glob
#import moviepy.editor as mpy

BATCH_SIZE = 16
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"


def predict():
    shape = (304, 228)
    img = cv2.imread('image5.jpg')
    cv2.imwrite('left_scene.jpg', img)
    shape_org = img.shape[:2]
    print(shape_org)
    img = cv2.resize(img, shape,interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img.astype(np.float32)
    inputs = np.zeros((1, 228, 304, 3), dtype='float32')
    inputs[0] = img_array

    with tf.Graph().as_default():
        '''
        with tf.device('/cpu:0'):
            batch_generator = BatchGenerator(batch_size=BATCH_SIZE)
            # train_images, train_depths, train_pixels_mask = batch_generator.csv_inputs(TRAIN_FILE)
            train_images, train_depths, train_pixels_mask, names = batch_generator.csv_inputs(TRAIN_FILE, batch_size=BATCH_SIZE)
            # Create a placeholder for the input image
            # input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))
        '''
        input = tf.placeholder(tf.float32, [None, 228, 304, 3], name='input_batch')
        # Construct the network
        predictions = build_model(input)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Evalute the network for the given image
            # initialize the queue threads to start to shovel data
            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(coord=coord)
            depth_pred = sess.run(predictions, feed_dict={input: inputs})

            # output_groundtruth(depth_pred, ground_truth, "data/predictions/predict_scale1_%05d_%05d" % (0, 0))
            depth_pred = np.reshape(depth_pred, (128, 160))
            depth_pred = (depth_pred / np.max(depth_pred)) * 255.0
            depth_pred = np.clip(depth_pred, 0, 255).astype(np.uint8)
            # print(depth_pred.shape)
            depth_map = Image.fromarray(depth_pred)
            depth_map.resize((304, 228))
            depth_map.save('depth.png')

            d = cv2.imread('depth.png', 0)
            d = cv2.resize(d, (shape_org[1],shape_org[0]))
            cv2.imwrite('depth_scaled.png', d)

            '''
            coord.request_stop()
            coord.join(threads)
            sess.close()
            '''


if __name__ == '__main__':
    predict()
    Z = cv2.imread('depth_scaled.png', 0)
    Z = Z.astype(np.float32)
    #Z = Z- np.min(Z)
    depth_levels = np.unique(Z)
    depth_levels = np.sort(depth_levels)
    print(depth_levels)

    f = np.median(Z)

    B = 5
    ds = (B * (Z - f)) // Z
    ds = ds.astype(np.int8)

    L = cv2.imread('image5.jpg')
    L = cv2.cvtColor(L, cv2.COLOR_BGR2RGB)
    L = L.astype(np.int8)

    print(L.shape)
    R = np.zeros_like(L)
    H, W = Z.shape
    print(H)
    print(W)
    for i in range(H):
        for j in range(W):
            dw = ds[i, j]
            new_j = j + dw
            if new_j >= W:
                new_j = W - 1
            if new_j < 0:
                new_j = 0
            R[i, j] = L[i, new_j]

    R = Image.fromarray(R, 'RGB')
    R.save('right_scene.jpg')

    #gif_name = 'scene_min'
    #fps = 12
    #file_list = glob.glob('*_scene.jpg')  # Get all the pngs in the current directory
    #clip = mpy.ImageSequenceClip(file_list, fps=fps)
    #clip.write_gif('{}.gif'.format(gif_name), fps=fps)
