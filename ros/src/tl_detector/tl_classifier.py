from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import datetime

class TLClassifier(object):
    def __init__(self, is_simulator):

        self.result_map = { 1: TrafficLight.GREEN, 2: TrafficLight.RED, 3: TrafficLight.YELLOW  }

        graph_path = self._get_file_path(is_simulator)
        #print(graph_path)

        self.graph = tf.Graph()
        self.threshold = .5

        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as gf:
                graph_def.ParseFromString(gf.read())
                tf.import_graph_def(graph_def, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')

        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):

        img_expand = np.expand_dims(image, axis=0)
        start = datetime.datetime.now()
        with self.graph.as_default():  
            (scores, classes) = self.sess.run(
                [self.scores, self.classes], feed_dict={self.image_tensor: img_expand} )
    
        tm = datetime.datetime.now() - start

        r_scores = np.squeeze(scores)
        r_classes = np.squeeze(classes).astype(np.int32)

        r_class = 4 if r_scores[0] < self.threshold else r_classes[0]

        return self.result_map.get(r_class, TrafficLight.UNKNOWN)

    def _get_file_path(self, is_simulator):
        if is_simulator:
            return 'model/simulator_graph.pb'
        else:
            return 'model/site_graph.pb'