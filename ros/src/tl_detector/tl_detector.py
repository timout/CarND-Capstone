#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 2
I_COUNT_CHECK = 4

VEHICLE_LEN = 5.

class WPControl(object):

    def __init__(self, config):

        self.stop_line_p = config['stop_line_positions'] 
        self.stop_line_wp_i = None # Stop line position waypoint indexes
        self.stop_line_tree = None #Stop line search tree

        self.waypoints = None
        self.waypoints_pos = None
        self.wp_tree = None # Waypoints search tree

    def waypoints_cb(self, waypoints):
        if not self.waypoints_pos:
            self.waypoints = waypoints
            self.waypoints_pos = self._waypoints_to_position(waypoints)
            self.wp_tree = KDTree(self.waypoints_pos)
            self.stop_line_wp_i = self._create_stop_line_wp_index_map()
            self.stop_line_tree = KDTree(self.stop_line_p)

    def _waypoints_to_position(self, waypoints):
        return [ [w.pose.pose.position.x, w.pose.pose.position.y] for w in waypoints.waypoints]

    def _create_stop_line_wp_index_map(self):
        return { i: self._update_stop_line_pos(i, pos) for i, pos in enumerate(self.stop_line_p) }

    def _update_stop_line_pos(self, i, pos):
        wp_i = self.get_closest_wp(pos)[1]
        yaw = self._get_yaw(wp_i)
        p = self._get_stop_line_pos(yaw, wp_i)
        self.stop_line_p[i] = p
        return self.get_closest_wp(p)[1]

    def _get_yaw(self, wp_i):
        if wp_i == 0: return 0
        wp = self.waypoints_pos[wp_i]
        wp_prev = self.waypoints_pos[wp_i-1]
        return math.atan2(( wp[1] - wp_prev[1] ), ( wp[0] - wp_prev[0] ) )

    def _get_stop_line_pos(self, yaw, wp_i):
        wp = self.waypoints_pos[wp_i]
        lx = wp[0] * math.cos(-yaw) - wp[1] * math.sin(-yaw)
        ly = wp[0] * math.sin(-yaw) + wp[1] * math.cos(-yaw)
        lx -= VEHICLE_LEN / 2
        x = lx * math.cos(yaw) - ly * math.sin(yaw)
        y = lx * math.sin(yaw) + ly * math.cos(yaw)
        return [x,y]

    def is_ready(self):
        return self.wp_tree is not None and self.stop_line_tree is not None

    def get_stop_line_wp_i(self, i):
        return self.stop_line_wp_i[i]

    def get_next_stop_line(self, sl_i):
        next_sl_i = sl_i + 1
        if next_sl_i == len(self.stop_line_p): next_sl_i = 0
        next_sl_wp_i = self.get_stop_line_wp_i(next_sl_i)
        return next_sl_i, next_sl_wp_i

    def get_closest_stop_line(self, position): 
        """ Get closest waypoint index to position """
        # TODO: find next one, ignore previous
        p = [position.x, position.y]
        return self.stop_line_tree.query(p, 1)

    def get_closest_wp(self, pos): 
        """ Get closest waypoint index to position """
        return self.wp_tree.query(pos, 1)

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.config = yaml.load(rospy.get_param("/traffic_light_config"))
        self.is_simulator = not self.config["is_site"]
        print("Is Simulator %d" % self.is_simulator)
        
        #self.process_image_fn = self.process_image_test()
        self.process_image_fn = self.process_image_sim if self.is_simulator else self.process_image

        self.wp_control = WPControl(self.config)

        self.light_classifier = TLClassifier(self.is_simulator)

        self.pose = None
        self.camera_image = None
        self.lights = []
		
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.wp_control.waypoints_cb)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb)
        
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        
        #self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_wp_i = -1
        self.state_count = 0

        self.stop_line_i = 0 #test
        
        self.i_count = -1

        self.start_classification = False

        self.check_count = I_COUNT_CHECK
        self.sl_distance = 1000
        self.diff = -10

        self.loop()


    def loop(self):
        rate = rospy.Rate(10)
        while not self.wp_control.is_ready():
            print("is not ready")
            rate.sleep()

        while not rospy.is_shutdown():
            '''Publish upcoming red lights at camera frequency.'''
            if self.pose is not None:
                light_wp_i = self.get_next_traffic_light()
                if not self.start_classification:
                    self.state_count = 0
                    self.last_wp_i = -1
                    self.upcoming_red_light_pub.publish(Int32(self.last_wp_i))
                else:
                    #print("SP:Light WP Index: ",light_wp_i, "Traffic Light: ", self.state)

                    if self.state_count >= STATE_COUNT_THRESHOLD:
                        if self.state != TrafficLight.RED: light_wp_i = -1                               
                        self.last_wp_i = light_wp_i
                        self.upcoming_red_light_pub.publish(Int32(self.last_wp_i))
                    else:
                        self.upcoming_red_light_pub.publish(Int32(self.last_wp_i))
                      
                    print("SP: State: ", self.state, " Count: ", self.state_count, " Last wp i:" , self.last_wp_i, " stop line i: ", self.stop_line_i)            

            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg


    def traffic_cb(self, msg):
        self.lights = msg.lights


    def image_cb(self, msg):
        """ Camera images receiver """
        self.camera_image = msg
        if not self.start_classification: 
            self.state = TrafficLight.UNKNOWN
            self.i_count = -1
        else: 
            self.process_image_fn()


    def process_image_test(self):
        cur_img_state = self.get_test_light_state() 
        if self.state != cur_img_state:
            self.state_count = 0
            self.state = cur_img_state
        else:
            self.state_count += 1    


    def process_image_sim(self):
        self.i_count += 1
        #process every `self.check_count` image 
        if ( self.i_count % self.check_count == 0 ) :                    
            cur_img_state = self.classify_light_image()
            self.process_classified_image_state(cur_img_state)


    def process_image(self):                  
        cur_img_state = self.classify_light_image()
        self.process_classified_image_state(cur_img_state)
  

    def classify_test_light_image(self):
        """ Determines the current color of the traffic light """
        if( len(self.lights) == 0  ): return TrafficLight.UNKNOWN
        # test detection: /vehicle/traffic_lights topic
        return self.lights[self.stop_line_i].state  


    def classify_light_image(self):
        """ Determines the current color of the traffic light  """
        if( self.camera_image is None ): return TrafficLight.UNKNOWN
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        # Classification
        return self.light_classifier.get_classification(cv_image) 


    def process_classified_image_state(self, cur_img_state):
        """ Process classified light image state to find current state"""
        self.print_img_color(cur_img_state)
        #default is red
        if self.state == TrafficLight.UNKNOWN and cur_img_state == TrafficLight.UNKNOWN:
            self.state = TrafficLight.RED
            cur_img_state = TrafficLight.RED
        elif self.state != TrafficLight.UNKNOWN and cur_img_state == TrafficLight.UNKNOWN:
            cur_img_state = self.state

        if self.state != cur_img_state:
            self.state_count = 0
            self.state = cur_img_state
        else:
            self.state_count += 1  

    def print_img_color(self, cur_img_state):
        r_map = { TrafficLight.GREEN: "Green", TrafficLight.RED: "Red", TrafficLight.YELLOW: "Yellow" }
        print("Classification Result: ", r_map.get(cur_img_state, "Unknown"))


    def get_next_traffic_light(self):
        """ Finds closest visible traffic light, if one exists """
        current_wp_i = self.wp_control.get_closest_wp([self.pose.pose.position.x, self.pose.pose.position.y])[1]

        self.sl_distance, self.stop_line_i = self.wp_control.get_closest_stop_line(self.pose.pose.position)

        stop_line_wp_i = self.wp_control.get_stop_line_wp_i(self.stop_line_i)

        self.diff = stop_line_wp_i - current_wp_i

        #print("Closst light Diff: ", self.diff, " stop line i:", self.stop_line_i, " sl-distance: ", self.sl_distance)
        self.start_classification = ( self.diff > -5 and self.sl_distance < 50 )

        return stop_line_wp_i


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
