#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import TwistStamped
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import statistics
from collections import deque
import time
import NashNet

class GameTheoryRacingNode(Node):
    def __init__(self):

        super().__init__('racing_controller')

        ### used to calculate reading median to avoid hallucinations
        self.history_tb3_1 = deque(maxlen=3)  # Only need LiDAR history for Robot 2 now

        ### current distances and state
        self.dist_tb3_1 = float('inf')
        self.angle_tb3_1 = 0.0

        self.err = 0.0
        self.detected = False

        self.bridge = CvBridge()

        ### subscribe to sensors
        # Robot 1 uses the Camera (Leader)
        self.sub_image_tb3 = self.create_subscription(
            Image, '/tb3/image_raw', self.image_cb_tb3, 10)

        # Robot 2 uses the LiDAR (Follower)
        self.sub_tb3_1 = self.create_subscription(
            LaserScan, '/tb3_1/scan', self.scan_cb_tb3_1, qos_profile_sensor_data)

        ### subscribe to motors
        self.pub_tb3 = self.create_publisher(TwistStamped, '/tb3/cmd_vel', 10)
        self.pub_tb3_1 = self.create_publisher(TwistStamped, '/tb3_1/cmd_vel', 10)

        # Evaluates the matrices and updates motors 10 times a second
        self.decision_timer = self.create_timer(0.1, self.game_loop)

        self.get_logger().info("Racing Controller Started! Waiting for sensor")

        self.prev_angle_tb3_1 = 0.0
        self.angle_rate = 0.0

        self.merged = False
        self.merge_timer = time.time()  # start immediately, not None
        self.MERGE_DURATION = 3.3 # tune this,,,increase if it hasn't reached the lane yet

    def image_cb_tb3(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w, _ = cv_image.shape

        roi = cv_image[int(h * 0.85):int(h * 0.95), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([101, 77, 73])
        upper_blue = np.array([139, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        ## center between two lanes logic

        ys, xs = np.where(mask > 0)
        self.err = 0.0
        roi_w = mask.shape[1]

        if len(xs) > 0:
            mid = roi_w // 2

            left_points = xs[xs < mid]
            right_points = xs[xs > mid]

            if len(left_points) > 0 and len(right_points) > 0:
                left_center = np.mean(left_points)
                right_center = np.mean(right_points)
                lane_center = ((left_center + right_center) / 2)
                self.err = lane_center - roi_w / 2
                self.detected = True
            else:
                self.detected = False
        else:
            self.detected = False

        if self.detected:
            err_wiggle_room = 4
            if abs(self.err) < err_wiggle_room:
                self.err = 0

    def scan_cb_tb3_1(self, msg):
        self.dist_tb3_1, self.angle_tb3_1 = self.process_scan(msg, self.history_tb3_1)

    def process_scan(self, msg, history_buffer):
        ### reject any nan or ZERO values
        closest_dist = float('inf')
        closest_index = -1

        for i, r in enumerate(msg.ranges):
            if not math.isnan(r) and 0.12 < r < 3.5:
                if r < closest_dist:
                    closest_dist = r
                    closest_index = i

        if closest_index == -1:
            return float('inf'), 0.0

        angle_rad = msg.angle_min + (closest_index * msg.angle_increment)

        ### Normalize the ts angle to be between -pi and pi
        if angle_rad > math.pi:
            angle_rad -= 2 * math.pi

        history_buffer.append(closest_dist)
        if len(history_buffer) < 3:
            return closest_dist, angle_rad

        smoothed_distance = statistics.median(history_buffer)

        return smoothed_distance, angle_rad

    def game_loop(self):
        if len(self.history_tb3_1) == 0:
            return  ## not enough data yet
        self.angle_rate = self.angle_tb3_1 - self.prev_angle_tb3_1
        speedLane, speedMerger = NashNet.get_speeds(self.dist_tb3_1)
        self.prev_angle_tb3_1 = self.angle_tb3_1
        self.get_logger().info(
            f"Line Found: {self.detected} | Opponent Dist: {self.dist_tb3_1:.2f}m | Angle: {self.angle_tb3_1:.2f}rad",
            throttle_duration_sec=0.5
        )

        ### Create empty velocity messages
        cmd_tb3 = TwistStamped()
        cmd_tb3.header.stamp = self.get_clock().now().to_msg()
        cmd_tb3.header.frame_id = 'base_link'

        cmd_tb3_1 = TwistStamped()
        cmd_tb3_1.header.stamp = self.get_clock().now().to_msg()
        cmd_tb3_1.header.frame_id = 'base_link'

        ### leader camera tb3
        if self.detected:
            print("forward")
            cmd_tb3.twist.linear.x = speedLane
            cmd_tb3.twist.angular.z = -(float(self.err) * 0.0022)
        else:
            cmd_tb3.twist.linear.x = 0.0
            cmd_tb3.twist.angular.z = 0.0

        ### merger lidar tb3_1
        if self.dist_tb3_1 < float('inf'):
            # Transition: straight line approach for fixed duration
            if not self.merged:
                if self.merge_timer is None:
                    self.merge_timer = time.time()
                elif time.time() - self.merge_timer > self.MERGE_DURATION:
                    self.merged = True

            if not self.merged:
                # APPROACH: Drive straight into the lane
                # tb3 starts behind/beside us at ~45deg, just go forward
                sample_speed = speedMerger
                sample_steering = 0.0
            elif abs(self.angle_tb3_1) > (math.pi / 2):
                # IN LANE - FOLLOW MODE (tb3 is behind us)
                # Use tb3's angle to stay aligned with the lane
                if self.angle_tb3_1 > 0:
                    angle_to_back = self.angle_tb3_1 - math.pi
                else:
                    angle_to_back = self.angle_tb3_1 + math.pi
                sample_speed = speedMerger
                sample_steering = -((angle_to_back * 0.8) + (self.angle_rate * 0.5))
            else:
                # PACE CAR MODE (tb3 caught up and is now in front)
                sample_steering = (self.angle_tb3_1 * 0.8)
                if self.dist_tb3_1 < 0.5:
                    sample_speed = speedMerger
                else:
                    sample_speed = -speedLane
            cmd_tb3_1.twist.linear.x = sample_speed
            cmd_tb3_1.twist.angular.z = sample_steering
        self.pub_tb3.publish(cmd_tb3)
        self.pub_tb3_1.publish(cmd_tb3_1)


def main(args=None):
    rclpy.init(args=args)
    node = GameTheoryRacingNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        print("\n[Shutting Down] Initializing Emergency Brake...")
    finally:
        ###  stop
        if rclpy.ok():
            try:
                stop_cmd = TwistStamped()
                stop_cmd.twist.linear.x = 0.0
                stop_cmd.twist.angular.z = 0.0
                node.pub_tb3.publish(stop_cmd)
                node.pub_tb3_1.publish(stop_cmd)
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()