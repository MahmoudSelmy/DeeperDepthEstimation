from tensorflow.python.platform import gfile
import numpy as np
from PIL import Image

def output_predict(depths, images, output_dir):
    print("output predict into %s" % output_dir)
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)
    for i, (image, depth) in enumerate(zip(images, depths)):
        pilimg = Image.fromarray(np.uint8(image))
        image_name = "%s/%05d_org.png" % (output_dir, i)
        pilimg.save(image_name)

        depth = depth.transpose(2, 0, 1)
        if np.max(depth) != 0:
            ra_depth = (depth/np.max(depth))*255.0
        else:
            ra_depth = depth*255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
        depth_name = "%s/%05d.png" % (output_dir, i)
        depth_pil.save(depth_name)

def output_groundtruth(depths, grounds, output_dir):
    print("output predict into %s" % output_dir)
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)
    for i, (ground, depth) in enumerate(zip(grounds, depths)):

        ground = ground.transpose(2, 0, 1)
        if np.max(ground) != 0:
            ra_ground = (ground / np.max(ground)) * 255.0
        else:
            ra_ground = ground * 255.0
        depth_pil = Image.fromarray(np.uint8(ra_ground[0]), mode="L")
        depth_name = "%s/%05d_ground.png" % (output_dir, i)
        depth_pil.save(depth_name)

        depth = depth.transpose(2, 0, 1)
        if np.max(depth) != 0:
            ra_depth = (depth/np.max(depth))*255.0
        else:
            ra_depth = depth*255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
        depth_name = "%s/%05d.png" % (output_dir, i)
        depth_pil.save(depth_name)