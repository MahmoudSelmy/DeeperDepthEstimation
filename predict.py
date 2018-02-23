import numpy as np
import tensorflow as tf
from Utills import output_groundtruth
from PIL import Image
from model import build_model
from data_preprocessing import BatchGenerator

BATCH_SIZE = 16
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"


def predict():

    with tf.Graph().as_default():

        with tf.device('/cpu:0'):
            batch_generator = BatchGenerator(batch_size=BATCH_SIZE)
            # train_images, train_depths, train_pixels_mask = batch_generator.csv_inputs(TRAIN_FILE)
            train_images, train_depths, train_pixels_mask, names = batch_generator.csv_inputs(TRAIN_FILE, batch_size=BATCH_SIZE)
            # Create a placeholder for the input image
            # input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

        # Construct the network
        predictions = build_model(train_images)



        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Evalute the network for the given image
            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            batch_images, ground_truth, depth_pred = sess.run([train_images, train_depths, predictions])

            output_groundtruth(depth_pred, ground_truth, "data/predictions/predict_scale1_%05d_%05d" % (0, 0))

            coord.request_stop()
            coord.join(threads)
            sess.close()



def main(argv=None):
    predict()

if __name__ == '__main__':
    tf.app.run()
