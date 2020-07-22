'''
You should not edit helper.py as part of your submission.
This file is used primarily to download vgg if it has not yet been,
give you the progress of the download, get batches for your training,
as well as around generating and saving the image outputs.
'''
from PIL import Image
import re
import numpy as np
import os
from glob import glob
from urllib.request import urlretrieve
ORIG_DATA_DIR = "../data_road/"#
# POST_PROCESSING_DATA_DIR = "../data_dir/"#


def data_preprocessing_and_load(image_shape,dataset_type = "training"):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    data_folder = os.path.join(ORIG_DATA_DIR, dataset_type)
    # save_folder = os.path.join(POST_PROCESSING_DATA_DIR, dataset_type)

    # Grab image and label paths
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
    background_color = np.array([255, 0, 0])
    count = 0
    # save_img = os.path.join(save_folder,"image_2")
    # save_gt = os.path.join(save_folder,"gt_image_2")
    # os.makedirs(save_img,exist_ok=True)
    # os.makedirs(save_gt,exist_ok = True)
    lhs = []
    rhs = []


    for image_path in image_paths:

        gt_image_file = label_paths[os.path.basename(image_path)]
        # Re-size to image_shape
        image_import = Image.open(image_path)
        gt_image_import = Image.open(gt_image_file)
        image_import = image_import.resize(image_shape,Image.BICUBIC)
        gt_image_import = gt_image_import.resize(image_shape,Image.BICUBIC)


        image = np.array(image_import)
        gt_image = np.array(gt_image_import)

        # Create "one-hot-like" labels by class
        gt_bg = np.all(gt_image == background_color, axis=2)
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)




        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
        gt_image = gt_image.astype('uint8')
        # image_path = os.path.join(save_img, "{}.png".format(str(count_image)) )
        # gt_image_path = os.path.join(save_gt,"{}.png".format(str(count_image)))
        lhs.append(image)
        rhs.append(gt_image)
        print(count)
        count+=1
    lhs= np.asarray(lhs)
    rhs = np.asarray(rhs)

    return lhs, rhs
        




def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep probability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        # Run inference
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        # Splice out second column (road), reshape output back to image_shape
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        # If road softmax > 0.5, prediction is road
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        # Create mask based on segmentation to apply to original image
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    """
    Save test images with semantic masks of lane predictions to runs_dir.
    :param runs_dir: Directory to save output images
    :param data_dir: Path to the directory that contains the datasets
    :param sess: TF session
    :param image_shape: Tuple - Shape of image
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep probability
    :param input_image: TF Placeholder for the image placeholder
    """
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)



if __name__ == "__main__":
    data_preprocessing_and_load((160, 576))
