# -*- coding: utf-8 -*-
import sys
from utils import *
from constants import *
from process_data import process_data
from generator import generator
from discriminator import discriminator, lrelu
from test import test 
from resize_img import resize_img

def train(img_resized, img_generated,_saved_models):
    
    with tf.variable_scope('input'):
        #real and fake image placholders
        real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')
    
    # gan
    fake_image = generator(random_input, random_dim, is_train)
    
    real_result = discriminator(real_image, is_train)
    fake_result = discriminator(fake_image, is_train, reuse=True)
    
    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
    g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.
            

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)
   
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    batch_size = BATCH_SIZE
    image_batch, samples_num = process_data(img_resized)
    
    batch_num = int(samples_num / batch_size)
    total_batch = 0
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    # continue training
    save_path = saver.save(sess, "/tmp/model.ckpt")
    ckpt = tf.train.latest_checkpoint(_saved_models)
    saver.restore(sess, save_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print(' ---- total training sample num: %d' % samples_num)
    print(' ---- batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, EPOCH))
    print(' ---- start training...')
    for i in range(EPOCH):
        print(" ---- Running epoch {}/{}...".format(i, EPOCH))
        for j in range(batch_num):
            print(j)
            d_iters = 5
            g_iters = 1

            train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            for k in range(d_iters):
                print(k)
                train_image = sess.run(image_batch)
                #gan clip weights
                sess.run(d_clip)
                
                # Update the discriminator
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})

            # Update the generator
            for k in range(g_iters):
                _, gLoss = sess.run([trainer_g, g_loss],
                                    feed_dict={random_input: train_noise, is_train: True})

            
        # save check point every 500 epoch
        if i%500 == 0:
            if not os.path.exists(_saved_models):
                os.makedirs(_saved_models)
            saver.save(sess, _saved_models + '/' + str(i))  
        if i%50 == 0:
            
            # save images
            if   os.path.exists(img_generated):
                shutil.rmtree(img_generated, ignore_errors=True)
            os.makedirs(img_generated)
            sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})

            # imgtest.astype(np.uint8)
            save_images(imgtest, [8,8] ,img_generated + '/epoch' + str(i) + '.jpg')
            
            print(' ---- train: [%d],d_loss: %f,g_loss: %f' % (i, dLoss, gLoss))
    coord.request_stop()
    coord.join(threads)




if __name__ == "__main__":
    
    if len (sys.argv) != 3 :
        print(" Args not found : global path for arg 1 and image subdirectory for arg 2.")
        sys.exit(1)
    if not os.path.exists(sys.argv[1]) or not os.path.exists(sys.argv[2]):
        print('----------------------------------------------------------------------------')
        print(' --- Error : Args is not a path or a direcotry ! ')
        sys.exit(1)
        
    if len(os.listdir(sys.argv[1])) == 0:
        print('----------------------------------------------------------------------------')
        print(' --- Error : Arg 1, images directory is empty ! ')
        sys.exit(1)
    print('----------------------------------------------------------------------------')   
    print(' --- Beginning Resize image: ')
    resize_img(sys.argv[1],sys.argv[2]+"/_resized",WIDTH,HEIGHT,CHANNEL)
    print('----------------------------------------------------------------------------')
    print(' --- Beginning tain model: ')
    train(sys.argv[2]+"/_resized",sys.argv[2]+"/_img_generated",sys.argv[2]+"/_saved_model")
    print(' --- End tain model: ')
    print('----------------------------------------------------------------------------')
    print(' --- Beginning test model: ')
    test(sys.argv[2]+"/_saved_model")
    print(' --- Beginning test model: ')

