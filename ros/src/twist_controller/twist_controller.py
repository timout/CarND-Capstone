from pid import PID
import rospy
from lowpass import LowPassFilter
from yaw_controller import YawController

ONE_MPH = 0.44704

class Controller(object):

    def __init__(self):

        self.mass = rospy.get_param('~vehicle_mass', 1736.35)
        self.fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        self.brake_deadband = rospy.get_param('~brake_deadband', .1)
        self.decel_limit = rospy.get_param('~decel_limit', -5)
        self.accel_limit = rospy.get_param('~accel_limit', 1.)
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        self.wheel_base = rospy.get_param('~wheel_base', 2.8498)
        self.steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        self.max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        self.max_steer_angle = rospy.get_param('~max_steer_angle', 8.)
        self.fuel_density = rospy.get_param('~fuel_density', 2.858 )

        self.vehicle_mass = self.mass + self.fuel_capacity * self.fuel_density
        self.vehicle_mass_wheel_radius = self.vehicle_mass * self.wheel_radius

        self.throttle_pid = PID(.4, .1, 0., 0., 1.)

        self.accel_pid = PID(2., 0., 0., self.decel_limit, self.accel_limit)

        self.accel_f = LowPassFilter(.5, 0.02)

        self.yaw_control = YawController(self.wheel_base, self.steer_ratio, 1, self.max_lat_accel, self.max_steer_angle)
        
        self.min_speed = 2.
        self.last_velocity = 0.0
        self.last_time = 0


    def control(self, linear_velocity, angular_velocity, current_velocity):
        delta_t = self.delta_time()
        lv_delta = linear_velocity - current_velocity
        v_delta = current_velocity - self.last_velocity
        self.last_velocity = current_velocity

        if delta_t == 0:
            self.reset()
            return 0., 0., 0.

        if abs(linear_velocity) < ONE_MPH: self.accel_pid.reset()

        accel_cmd = self.accel_pid.step(lv_delta, delta_t)

        if linear_velocity < .01:
            accel_cmd = min(accel_cmd, -530. / self.vehicle_mass / self.wheel_radius)
        elif linear_velocity < self.min_speed:
            angular_velocity *= self.min_speed / linear_velocity
            linear_velocity = self.min_speed
        
        throttle = self.get_trottle(accel_cmd, v_delta, delta_t)

        brake = self.get_brake(accel_cmd, linear_velocity)

        steering = self.yaw_control.get_steering(linear_velocity, angular_velocity, current_velocity)

        return throttle, brake, steering

    def delta_time(self):
        """ Calculate step time delta current time - previous time. """
        current_time = rospy.get_time()
        if self.last_time == 0: self.last_time = current_time
        delta_t = current_time - self.last_time
        self.last_time = current_time
        return delta_t

    def get_brake(self, accel_cmd, linear_velocity):
        if (accel_cmd < -self.brake_deadband) or (linear_velocity < self.min_speed):
            return -accel_cmd * self.vehicle_mass_wheel_radius
        return 0.0

    def get_trottle(self, accel_cmd, v_delta, delta_t):
        accel = v_delta / delta_t
        self.accel_f.filt(accel)
        if accel_cmd >= 0:
            return self.throttle_pid.step(accel_cmd - self.accel_f.get(), delta_t)
        self.throttle_pid.reset()
        return 0.0

    def reset(self):
        self.accel_pid.reset()
        self.throttle_pid.reset()

