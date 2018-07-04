import tensorflow as tf
from model import build_model
import time
import cv2
import numpy as np
from PIL import Image
from multiprocessing import Pool
import glob
import moviepy.editor as mpy
import os

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


def shift_with_angle(L, ds, angle_unit_vector, index):
    print(angle_unit_vector)
    Vh, Vw = angle_unit_vector
    R = np.zeros_like(L)
    H, W = ds.shape

    for i in range(H):
        for j in range(W):
            d = ds[i, j]

            new_i = i + d * Vh
            if new_i >= H:
                new_i = H - 1
            if new_i < 0:
                new_i = 0

            new_j = j + d * Vw
            if new_j >= W:
                new_j = W - 1
            if new_j < 0:
                new_j = 0

            R[i, j] = L[new_i, new_j]

    R = Image.fromarray(R, 'RGB')
    img_name = './shifts/' + str(index) + '_scene.jpg'
    R.save(img_name)


def shift_oriented_threaded(number_of_threads=8):
    L = cv2.imread('left_scene.jpg')
    L = cv2.cvtColor(L, cv2.COLOR_BGR2RGB)
    H, W, _ = L.shape
    L = L.astype(np.int8)

    Z = cv2.imread('depth.png', 0)
    print(W)
    print(H)
    Z = cv2.resize(Z, (W, H))
    cv2.imwrite('depth_scaled_mr.png', Z)
    Z = Z.astype(np.float32)

    shift_unit_vectors = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    # scene_indices = range(len(shift_unit_vector))
    B = 10

    depth_levels = np.unique(Z)
    depth_levels = np.sort(depth_levels)
    f = np.min(depth_levels)
    ds = (B * (Z - f)) // Z
    ds = ds.astype(np.int8)

    p = Pool(number_of_threads)

    zipped_args = []
    for i, v in enumerate(shift_unit_vectors):
        zipped_args.append((L, ds, v, i))
        # shift_with_angle(L, ds, v, i)
    p.starmap(shift_with_angle, zipped_args)

    '''

    for B in B_values:
        shift_with_angle(L,Z,B,shift_unit_vector)
    '''


def create_gif_from_scenes(num_scenes=8):
    scenes_names = []
    base_name = './shifts/'
    for i in range(num_scenes):
        scenes_names.append(base_name + str(i) + '_scenes.jpg')
    fps = 16
    # images_names = ['right_scene_multi_threaded.jpg', 'image5.jpg']
    gif_name = 'scene_min'
    os.chdir("shifts")
    files_list = glob.glob('*_scene.jpg')
    # files_list = [f for f in listdir("/shifts") if isfile(join("/shifts", f))]
    # files_list = map(append_parent,files_list)
    clip = mpy.ImageSequenceClip(files_list, fps=fps)
    clip.write_gif('{}.gif'.format(gif_name), fps=fps)



if __name__ == '__main__':
    # predict()
    start_time = time.time()
    shift_oriented_threaded(number_of_threads=8)
    # create_gif_from_scenes()
    # predict_multi_threaded(number_of_threads=5)
    # shift()
    print("--- %s seconds ---" % (time.time() - start_time))

