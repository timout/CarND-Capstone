#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import PoseStamped,TwistStamped

from twist_controller import Controller


class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        self.current_velocity = None
        self.linear_velocity = None
        self.angular_velocity = None

        self.dbw_enabled = False

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)

        rospy.Subscriber('/vehicle/steering_report', SteeringReport, self.velocity_cb)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_status_cb)

        self.ctrl = Controller()

        self.loop()

    def velocity_cb(self, msg):
        self.current_velocity = msg.speed

    def twist_cb(self, msg):
        self.linear_velocity = msg.twist.linear.x
        self.angular_velocity = msg.twist.angular.z

    def dbw_status_cb(self, msg):
        self.dbw_enabled = msg.data

    def is_ready(self):
        return self.current_velocity is not None and self.linear_velocity is not None and self.angular_velocity is not None

    def loop(self):
        rate = rospy.Rate(50)  
        while not rospy.is_shutdown():
            if self.is_ready():
                if not self.dbw_enabled:
                    self.ctrl.reset()
                else:
                    t, b, s = self.ctrl.control(self.linear_velocity, self.angular_velocity, self.current_velocity)
                    self.publish(t, b, s)

            rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
