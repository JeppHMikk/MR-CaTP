#!/usr/bin/env python3
# license removed for brevity
import rospy
from std_msgs.msg import String
import numpy as np
from nav_msgs.msg import Odometry
from mrs_msgs.msg import TrajectoryReference
from geometry_msgs.msg import _Pose
from geometry_msgs.msg import PoseWithCovariance
from geometry_msgs.msg import Point

def mrcatp():

    rospy.init_node('mrcatp', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    pub = rospy.Publisher('chatter', String, queue_size=10)

    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        mrcatp()
    except rospy.ROSInterruptException:
        pass