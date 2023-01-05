#!/usr/bin/python3

import rospy
from std_msgs.msg import Bool

def digit_stop():
    rospy.init_node('stop_msg')
    stop_pub = rospy.Publisher('/digit_start',Bool,queue_size=10)
    msg = False
    stop_pub.publish(msg)

if __name__=='__main__':
    try:
        digit_stop()
    except rospy.ROSInterruptException:
        pass