import tensorflow as tf
from model import build_model
import time
import cv2
import numpy as np
from PIL import Image
from multiprocessing import Pool
import glob
import moviepy.editor as mpy

# import glob
# import moviepy.editor as mpy

BATCH_SIZE = 16
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"


def predict():
    shape = (304, 228)
    img = cv2.imread('image5.jpg')
    cv2.imwrite('left_scene.jpg', img)
    shape_org = img.shape[:2]
    print(shape_org)
    img = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC)
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
            d = cv2.resize(d, (shape_org[1], shape_org[0]))
            cv2.imwrite('depth_scaled.png', d)

            '''
            coord.request_stop()
            coord.join(threads)
            sess.close()
            '''


def shift_portion(base_vertical_index, disparity_portion):
    L = cv2.imread('left_scene.jpg')
    L = cv2.cvtColor(L, cv2.COLOR_BGR2RGB)
    L = L.astype(np.int8)

    print(base_vertical_index)
    H, W = disparity_portion.shape
    R = np.ones((H, W, 3), dtype=np.int8)
    for i in range(H):
        for j in range(W):
            '''
            dw = disparity_portion[i, j]
            new_j = j + dw
            if new_j >= W:
                new_j = W - 1
            if new_j < 0:
                new_j = 0
            R[i, j] = L[i + base_vertical_index, new_j]
            '''
            dh = disparity_portion[i, j]
            new_i = i + dh
            if new_i >= H:
                new_i = H - 1
            if new_i < 0:
                new_i = 0
            R[i, j] = L[new_i, j + base_vertical_index]
    return R


def predict_multi_threaded(number_of_threads=5):
    Z = cv2.imread('depth_scaled.png', 0)
    Z = Z.astype(np.float32)
    # Z = Z- np.min(Z)
    depth_levels = np.unique(Z)
    depth_levels = np.sort(depth_levels)

    f = np.max(depth_levels)

    B = -2
    ds = (B * (Z - f)) // Z
    ds = ds.astype(np.int8)

    H, W = ds.shape
    portion_H = W // number_of_threads
    portions_sizes_list = np.arange(1, number_of_threads)
    portions_sizes_list = portions_sizes_list * portion_H
    '''
    if H % number_of_threads != 0:
        portions_sizes_list = np.append(portions_sizes_list,[H % number_of_threads])
    '''
    # print(ds.shape)
    disparity_portions = np.split(ds, portions_sizes_list, axis=1)

    zipped_disparity_portions = []

    for i, p in enumerate(disparity_portions):
        zipped_disparity_portions.append((i * portion_H, p))

    # print(">>>" + str(len(zipped_disparity_portions)))

    p = Pool(number_of_threads)
    right_scene_portions = p.starmap(shift_portion, zipped_disparity_portions)
    right_scene = right_scene_portions[0]

    for portion in right_scene_portions[1:]:
        right_scene = np.concatenate((right_scene, portion), axis=0)
    R = Image.fromarray(right_scene, 'RGB')
    R.save('right_scene.jpg')
    gif_name = 'scene_min'
    fps = 12
    images_names = ['right_scene_multi_threaded.jpg', 'image5.jpg']

    file_list = glob.glob('*_scene.jpg')
    clip = mpy.ImageSequenceClip(file_list, fps=fps)
    clip.write_gif('{}.gif'.format(gif_name), fps=fps)


def create_gif_from_scenes(num_scenes=8):
    scenes_names = []
    base_name = './shifts/'
    for i in range(num_scenes):
        scenes_names.append(base_name + str(i) + '_scenes.jpg')
    fps = 12
    #images_names = ['right_scene_multi_threaded.jpg', 'image5.jpg']
    gif_name = 'scene_min'
    file_list = glob.glob('./shift/*_scene.jpg')
    clip = mpy.ImageSequenceClip(file_list, fps=fps)
    clip.write_gif('{}.gif'.format(gif_name), fps=fps)


if __name__ == '__main__':
    # predict()
    start_time = time.time()
    predict_multi_threaded(number_of_threads=8)
    # shift()
    print("--- %s seconds ---" % (time.time() - start_time))
