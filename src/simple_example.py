"""
A simple example illustrating basic use of the Ground Texture SLAM system.
"""
from typing import List
import numpy
import ground_texture_slam
import os
import argparse
import cv2


def create_images() -> List[numpy.ndarray]:
    """
    Create a series of fake images. This is just a rectangle translating across the image.
    @return The list of images
    @rtype List[numpy.ndarray]
    """
    image_list = []
    for j in range(10):
        image = numpy.zeros((600, 800), dtype=numpy.uint8)
        image[200:260, j*10:100] = 255
        image_list.append(image)
    return image_list

def load_images(path=""):
    """
    Load images from the path given
    """
    images = []
    img_path = [path + i for i in os.listdir(path)]
    img_path.sort()
    for i in img_path:
        img = cv2.imread(i)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        images.append(img)
    return images

def parse_args():
    parser = argparse.ArgumentParser(description="Run GroundTextureSLAM with specified image path.")
    parser.add_argument(
        '--path', 
        type=str, 
        required=True, 
        help='Path to the directory containing images for SLAM.'
    )
    return parser.parse_args()

def create_vocabulary() -> None:
    """
    Build a vocabulary tree out of random noise descriptors and save it locally for later use by the
    SLAM system.
    """
    print('Building the vocabulary from random descriptors...')
    all_descriptors = []
    for _ in range(100):
        all_descriptors.append(numpy.random.randint(
            low=0, high=256, size=(500, 32), dtype=numpy.uint8))
    vocab_options = ground_texture_slam.BagOfWords.VocabOptions()
    vocab_options.descriptors = all_descriptors
    bag_of_words = ground_texture_slam.BagOfWords(vocab_options)
    bag_of_words.save_vocabulary('example_vocab_python.bow')
    print('\tDone!')


if __name__ == '__main__':

    args = parse_args()

    # Build a vocabulary to use later.
    create_vocabulary()
    # Set all the parameters of this fake SLAM system. The important ones are the camera intrinsic
    # matrix and pose, and the vocabulary for bag of words.
    print('Loading system...')
    options = ground_texture_slam.GroundTextureSLAM.Options()
    options.bag_of_words_options.vocab_file = 'example_vocab_python.bow'
    options.keypoint_matcher_options.match_threshold = 0.6
    # Normally, you want a sliding window so that successive images don't get compared for loop
    # closure. However, I am specifically allowing that here since there are only a few images.
    options.sliding_window = 10

    print('Adding camera matrix with parameters...')
    fx = 2195.70853
    fy = 2195.56073
    ppx = 820.7289683
    ppy = 606.242723
    
    camera_matrix = numpy.array([
        [fx, 0, ppx],
        [0, fy, ppy],
        [0, 0, 1]
    ])
    options.image_parser_options.camera_intrinsic_matrix = camera_matrix
    camera_pose = numpy.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.25],
        [0.0, 0.0, 0.0, 1.0]
    ])
    options.image_parser_options.camera_pose = camera_pose
    # Get the images "captured" by the robot.
    # Also, create some fake start pose info. This doesn't matter so set it to the origin. You could
    # also set it to a known start pose to align with any data you are comparing against.
    # Unlike C++, there is only one input type here - numpy arrays.

    # path = '/media/ajax/AJ/ground-slam/kitchen_test_sq/test_path1/seq0031/'
    # images = create_images()
    images = load_images(args.path)    
    start_pose = numpy.zeros((3,), dtype=numpy.float64)
    start_covariance = numpy.identity(3, dtype=numpy.float64)
    print('Loading system... SLAM')
    system = ground_texture_slam.GroundTextureSLAM(
        options, images[0], start_pose, start_covariance)
    print('Adding images...')
    # Now add each image. This would be done as images are received.
    for i in range(1, len(images)):
        system.insert_measurement(images[i])
    # Once done, get the optimized poses. This can be done incrementally after each image as well.
    # The results probably won't be any good, since this is a bunch of random images. But it
    # illustrates the point.
    pose_estimates = system.get_pose_estimates_matrix()
    print('Results:')
    for i in range(len(images)):
        x = pose_estimates[i, 0]
        y = pose_estimates[i, 1]
        t = pose_estimates[i, 2]
        print(F'({x:0.6f}, {y:0.6f}, {t:0.6f})')
    
    npSavefile = 'results.npy'
    print(f'Saving Pose Estimates as {npSavefile}')
    numpy.save(npSavefile,pose_estimates)
