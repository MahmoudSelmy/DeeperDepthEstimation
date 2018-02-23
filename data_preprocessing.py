import tensorflow as tf

IMAGE_HEIGHT = 304
IMAGE_WIDTH = 228
TARGET_HEIGHT = 160
TARGET_WIDTH = 128
VGG_MEAN = [103.939, 116.779, 123.68]
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

class BatchGenerator:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    '''
    this function takes the train.csv file and use tensorflow to prepare batches to you 
    we also used it to resize to required size

    '''

    def csv_inputs(self, csv_file_path,batch_size=4):
        # print(csv_file_path)
        # list all (image,depth) pairs names and shuffle them
        filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=True)
        # reader to read text files
        reader = tf.TextLineReader()
        # get examples names
        _ , data_examples = reader.read(filename_queue)
        # record csv data into tensors
        image_examples, depth_targets = tf.decode_csv(data_examples, [["path"], ["annotation"]])
        # images
        jpg = tf.read_file(image_examples)
        image = tf.image.decode_jpeg(jpg, channels=3)
        # image = vgg16_preprocess(image)
        # depth
        depth_png = tf.read_file(depth_targets)
        depth = tf.image.decode_png(depth_png, channels=1)
        depth = tf.cast(depth, tf.float32)
        depth = tf.div(depth, [255.0])
        # resize
        image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
        # nan depth values > zero
        invalid_depth = tf.sign(depth)
        # generate batch
        images, depths, invalid_depths,image_example = tf.train.batch(
            [image, depth, invalid_depth,image_examples],
            batch_size=batch_size,
            num_threads=4,
            capacity=50 + 3 * batch_size,
        )
        return images, depths, invalid_depths,image_example


def vgg16_preprocess(image, mean=VGG_MEAN):
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.reverse(image, axis=[-1])  # RGB to BGR

    image = tf.cast(image, dtype=tf.float32)
    image = tf.subtract(image, mean)

    return image

