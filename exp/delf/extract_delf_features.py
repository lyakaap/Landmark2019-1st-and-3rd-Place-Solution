import os
import time
import tqdm

import tensorflow as tf
import numpy as np

from google.protobuf import text_format
from delf import delf_config_pb2
from delf import feature_extractor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
CONFIG_PATH = 'exp/delf/pretrained/delf_config_example.pbtxt'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100


def MakeExtractor(sess, config, import_scope=None):
    tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING],
        config.model_path,
        import_scope=import_scope)
    import_scope_prefix = (
        import_scope + '/' if import_scope is not None else '')
    input_image = sess.graph.get_tensor_by_name('%sinput_image:0' %
                                                import_scope_prefix)
    input_score_threshold = sess.graph.get_tensor_by_name(
        '%sinput_abs_thres:0' % import_scope_prefix)
    input_image_scales = sess.graph.get_tensor_by_name(
        '%sinput_scales:0' % import_scope_prefix)
    input_max_feature_num = sess.graph.get_tensor_by_name(
        '%sinput_max_feature_num:0' % import_scope_prefix)
    boxes = sess.graph.get_tensor_by_name(
        '%sboxes:0' % import_scope_prefix)
    raw_descriptors = sess.graph.get_tensor_by_name(
        '%sfeatures:0' % import_scope_prefix)
    feature_scales = sess.graph.get_tensor_by_name(
        '%sscales:0' % import_scope_prefix)
    attention_with_extra_dim = sess.graph.get_tensor_by_name(
        '%sscores:0' % import_scope_prefix)
    attention = tf.reshape(attention_with_extra_dim,
                           [tf.shape(attention_with_extra_dim)[0]])

    locations, descriptors = feature_extractor.DelfFeaturePostProcessing(
        boxes, raw_descriptors, config)

    def ExtractorFn(image):
        return sess.run(
            [locations, descriptors, feature_scales, attention],
            feed_dict={
                input_image: image,
                input_score_threshold: config.delf_local_config.score_threshold,  # noqa
                input_image_scales: list(config.image_scales),
                input_max_feature_num: config.delf_local_config.max_feature_num
            })

    return ExtractorFn


def main(csv_file, image_base_path, output_path):
    image_paths = []
    with open(csv_file, 'r') as f:
        f.readline()
        for line in f:
            id_ = line.split(',')[0]
            image_paths.append(
                image_base_path + '/' +
                id_[0] + '/' + id_[1] + '/' + id_[2] + '/' +
                id_ + '.jpg')
    num_images = len(image_paths)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('done! Found %d images', num_images)

    # Parse DelfConfig proto.
    config = delf_config_pb2.DelfConfig()
    with tf.gfile.FastGFile(CONFIG_PATH, 'r') as f:
        text_format.Merge(f.read(), config)

    # Create output directory if necessary.
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Reading list of images.
        filename_queue = tf.train.string_input_producer(
            image_paths, shuffle=False)
        reader = tf.WholeFileReader()
        _, value = reader.read(filename_queue)
        image_tf = tf.image.decode_jpeg(value, channels=3)

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            extractor_fn = MakeExtractor(sess, config)

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            start = time.clock()
            for i in tqdm.tqdm(range(num_images)):
                # Write to log-info once in a while.
                if i == 0:
                    tf.logging.info(
                        'Starting to extract DELF features from images...')
                elif i % _STATUS_CHECK_ITERATIONS == 0:
                    elapsed = (time.clock() - start)
                    tf.logging.info(
                        'Processing image %d out of %d, last %d '
                        'images took %f seconds', i, num_images,
                        _STATUS_CHECK_ITERATIONS,
                        elapsed)
                    start = time.clock()

                # # Get next image.
                im = sess.run(image_tf)

                # If descriptor already exists, skip its computation.
                out_desc_filename = os.path.splitext(os.path.basename(
                    image_paths[i]))[0] + '.npz'
                out_desc_fullpath = os.path.join(
                    output_path, out_desc_filename)
                if tf.gfile.Exists(out_desc_fullpath):
                    tf.logging.info('Skipping %s', image_paths[i])
                    continue

                # Extract and save features.
                (locations_out, descriptors_out, feature_scales_out,
                 attention_out) = extractor_fn(im)

                np.savez(
                    out_desc_fullpath,
                    loc=locations_out,
                    desc=descriptors_out.astype(np.float32),
                    feat=feature_scales_out)

            # Finalize enqueue threads.
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    main(ROOT + 'input/train.csv',
         ROOT + 'input/train',
         ROOT + 'input/delf/train/')
    main(ROOT + 'input/test19.csv',
         ROOT + 'input/test19',
         ROOT + 'input/delf/test19/')
