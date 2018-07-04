import time
import cv2
import numpy as np
from PIL import Image
from multiprocessing import Pool
import glob
import os
import moviepy.editor as mpy


def shift_portion(base_vertical_index, disparity_portion):
    print('open')
    L = cv2.imread('left_scene.jpg')
    L = cv2.cvtColor(L, cv2.COLOR_BGR2RGB)
    L = L.astype(np.int8)
    print('close')
    print(base_vertical_index)
    H, W = disparity_portion.shape
    R = np.ones((H, W, 3), dtype=np.int8)
    for i in range(H):
        for j in range(W):
            dw = disparity_portion[i, j]
            new_j = j + dw
            if new_j >= W:
                new_j = W - 1
            if new_j < 0:
                new_j = 0
            R[i, j] = L[i + base_vertical_index, new_j]
    return R


def predict_multi_threaded(number_of_threads=5):
    Z = cv2.imread('depth_scaled.png', 0)
    Z = Z.astype(np.float32)
    # Z = Z- np.min(Z)
    depth_levels = np.unique(Z)
    depth_levels = np.sort(depth_levels)

    f = np.median(depth_levels)

    B = 5
    ds = (B * (Z - f)) // Z
    ds = ds.astype(np.int8)

    H, W = ds.shape
    portion_H = H // number_of_threads
    portions_sizes_list = np.arange(1, number_of_threads)
    portions_sizes_list = portions_sizes_list * portion_H
    '''
    if H % number_of_threads != 0:
        portions_sizes_list = np.append(portions_sizes_list,[H % number_of_threads])
    '''
    # print(ds.shape)
    disparity_portions = np.split(ds, portions_sizes_list, axis=0)

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
    R.save('right_scene_multi_threaded.jpg')


def shift():
    Z = cv2.imread('depth_scaled.png', 0)
    Z = Z.astype(np.float32)
    # Z = Z- np.min(Z)
    depth_levels = np.unique(Z)
    depth_levels = np.sort(depth_levels)

    f = np.median(depth_levels)

    B = 5
    ds = (B * (Z - f)) // Z
    ds = ds.astype(np.int8)

    L = cv2.imread('image5.jpg')
    L = cv2.cvtColor(L, cv2.COLOR_BGR2RGB)
    L = L.astype(np.int8)

    H, W = Z.shape

    R = np.ones((H, W, 3), dtype=np.int8)
    print(R.shape)
    print(L.shape)
    for i in range(H):
        for j in range(W):
            dw = ds[i, j]
            new_j = j + dw
            if new_j >= W:
                new_j = W - 1
            if new_j < 0:
                new_j = 0
            R[i, j] = L[i, new_j]
    right = Image.fromarray(R, 'RGB')
    right.save('right_scene_head.jpg')


def shift_with_angle(L,ds, angle_unit_vector, index):
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
    start_time = time.time()
    shift_oriented_threaded(number_of_threads=8)
    # create_gif_from_scenes()
    # predict_multi_threaded(number_of_threads=5)
    # shift()
    print("--- %s seconds ---" % (time.time() - start_time))
