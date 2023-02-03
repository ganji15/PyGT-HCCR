import random
import glob
from preprocess.offline_to_graph.construct_graph import image2graph


SAMPLE_SET_NAME = 'chars'
img_dir = './samples/%s/*.png'%SAMPLE_SET_NAME
images = glob.glob(img_dir)
random.shuffle(images)
for i, image in enumerate(images):
    print(image)
    graph = image2graph(image, debug=True)
