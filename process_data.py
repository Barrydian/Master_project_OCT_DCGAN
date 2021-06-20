
from constants import *
 
def process_data(target_data):   
    #current_dir = os.getcwd()
    #oct_dir = os.path.join(current_dir, target_data)
    images = []
    for each in os.listdir(target_data):
        images.append(os.path.join(target_data,each))
    # print images    
    all_images = tf.convert_to_tensor(images, dtype = tf.string)
    
    images_queue = tf.train.slice_input_producer(
                                        [all_images])
                                        
    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels = CHANNEL)
    image = tf.image.random_brightness(image, max_delta = 0.1)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT,WIDTH,CHANNEL])

    # print image.get_shape()
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    
    iamges_batch = tf.train.shuffle_batch(
                                    [image], batch_size = BATCH_SIZE,
                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE,
                                    min_after_dequeue = 200)
    num_images = len(images)

    return iamges_batch, num_images

