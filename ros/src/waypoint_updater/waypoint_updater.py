#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane
from std_msgs.msg import Int32
from scipy.spatial import KDTree

import copy

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. 

class WPControl(object):

    def __init__(self):

        self.waypoints = None
        self.waypoints_pos = None
        self.wp_tree = None

    def waypoints_cb(self, waypoints):
        if not self.waypoints_pos:
            self.waypoints = waypoints
            self.waypoints_pos = [ [w.pose.pose.position.x, w.pose.pose.position.y] for w in waypoints.waypoints]
            # KDTree for waypoints search 
            self.wp_tree = KDTree(self.waypoints_pos)

    def is_ready(self):
        return self.wp_tree is not None

    def waypoint(self, i):
        return self.waypoints.waypoints[i]   

    # Next waypoint ahead of the vehicle
    def next_waypoint_i(self, pose):       
        position = [pose.position.x, pose.position.y]
        current_wp_i = self.wp_tree.query(position, 1)[1] 
        return current_wp_i


    def copy_wps(self, start_i, count):
        last_i = min(len(self.waypoints.waypoints), start_i + count)
        lane = Lane()
        lane.header = self.waypoints.header
        lane.waypoints = copy.deepcopy(self.waypoints.waypoints[start_i : last_i])
        return lane

class WaypointUpdater(object):

    def __init__(self):
                
        rospy.init_node('waypoint_updater')

        self.wp_control = WPControl()

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.wp_control.waypoints_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.vehicle = None

        self.next_light_i = -1
        self.vehicle_velocity = 0 

        self.loop()


    def pose_cb(self, msg):
        self.vehicle = msg

    def traffic_cb(self, msg):
        self.next_light_i = msg.data

    def velocity_cb(self, velocity):
        self.vehicle_velocity = velocity.twist.linear.x


    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.vehicle and self.wp_control.is_ready() and self.vehicle_velocity:
                start_i = self.wp_control.next_waypoint_i(self.vehicle.pose) # next waypoint to vehicle
                if self.next_light_i != -1:
                    step_count = self.next_light_i - start_i
                    lane = self.wp_control.copy_wps(start_i, step_count)
                    self.decelerate(lane.waypoints, start_i)
                else:
                    lane = self.wp_control.copy_wps(start_i, LOOKAHEAD_WPS)
                self.final_waypoints_pub.publish(lane)

            rate.sleep()

 
    def decelerate(self, waypoints, start_i):
        step_count = len(waypoints) - 1
        stop_i = step_count - 2
        target_speed = self.vehicle_velocity
        velocity_decrement = target_speed / step_count if step_count > 0 else target_speed
        for i, wp in enumerate(waypoints):
            target_speed -= velocity_decrement
            if target_speed <= 1 or i >= stop_i: target_speed = 0
            wp.twist.twist.linear.x = target_speed #set waypoint target speed


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
