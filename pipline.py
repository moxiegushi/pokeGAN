#coding=utf-8

import tensorflow as tf
slim  = tf.contrib.slim
import numpy as np
import os
import random


parent_dir = ''


def pipline(height, width, channel, batch_size, k_fold, no_k):
    dir_list = [parent_dir + 'Type_1', parent_dir + 'Type_2', parent_dir + 'Type_3']
    images = []
    labels = []
    label_count = 0
    for dir in dir_list:
        for img in os.listdir(dir):
            images.append(os.path.join(parent_dir, dir + '/' + img))
        # images = images + os.listdir(dir)
        labels = labels + [label_count] * len(os.listdir(dir))
        label_count = label_count + 1

    length = len(images)
    num_test = int(length / k_fold)
    num_train = length - num_test

    all_images = tf.convert_to_tensor(images)
    all_labels = tf.convert_to_tensor(labels)
    partition = [0] * length
    partition[no_k : no_k + num_test] = [1] * num_test

    # partition[:int(length/k_fold)] = [1] * int(length/k_fold)
    # random.shuffle(partition)

    images_train, images_test = tf.dynamic_partition(all_images, partition, 2)
    labels_train, labels_test = tf.dynamic_partition(all_labels, partition, 2)

    train_queue = tf.train.slice_input_producer(
                                                [images_train, labels_train],
                                                shuffle = False)
    
    file_content = tf.read_file(train_queue[0])
    train_image = tf.image.decode_jpeg(file_content)
    train_label = train_queue[1]
    # augment
    # train_image = tf.random_crop(train_image, [crop_size, crop_size, channel])
    train_image = tf.image.random_flip_up_down(train_image)
    train_image = tf.image.random_flip_left_right(train_image)
    train_image = tf.image.random_brightness(train_image, max_delta = 0.2)
    train_image = tf.image.resize_images(train_image, [height, width])
    train_image = tf.cast(train_image, tf.float32)
    train_image = train_image /255.0
    train_image = train_image * 2 - 1.0
    train_image.set_shape([height, width, channel])

    batch_images_train, batch_labels_train = tf.train.shuffle_batch(
                                                [train_image, train_label],
                                                batch_size = batch_size, num_threads = 4,
                                                capacity = 2000 + 3*batch_size,
                                                min_after_dequeue = 2000)


    test_queue = tf.train.slice_input_producer(
                                                [images_train, labels_train],
                                                shuffle = False)
    test_image = tf.image.decode_jpeg(tf.read_file(test_queue[0]))
    test_image = tf.image.resize_images(test_image, [height, width, channel])
    test_image = 2*(test_image/255.0) -1
    test_image.set_shape([height, width, channel])
    test_label = test_queue[1]

    batch_images_test, batch_labels_test = tf.train.batch([test_image, test_label],
                                                            batch_size = num_test)

    return batch_images_train, batch_labels_train, batch_images_test, batch_labels_test
