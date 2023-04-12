#!/usr/bin/python

from __future__ import print_function
import rospy
from std_msgs.msg import UInt16
import roslib;

import numpy as np
import math
import mobile_base



CHEST_LIGHT_FRAMERATE = 5.

class Command:
    def __init__(self, pan=0, tilt=0, eyes=0):
        self.pan = pan
        self.tilt = tilt
        self.eyes = eyes
    
    def updateState(self, additions):
        assert type(additions) == type(self)
        self.changePan(additions.pan)
        self.changeTilt(additions.tilt)
        self.changeEyes(additions.eyes)
        print(self.pan, "\t", self.tilt, "\t", self.eyes)

    def changePan(self, add):
        self.pan = self.bound(self.pan + add, -0.63, 0.78)

    def changeTilt(self, add):
        self.tilt = self.bound(self.tilt + add, -0.92, 0.29)

    def changeEyes(self, add):
        self.eyes = self.bound(self.eyes + add, -0.16, 0.41)

    def bound(self, n, minn, maxn):
        return max(min(maxn, n), minn)


class HeadTeleop:
    """
        mobile_base.HeadClient.PAN_LEFT = 0.78, 
		mobile_base.HeadClient.PAN_NEUTRAL = 0.0,
		mobile_base.HeadClient.PAN_RIGHT = -0.78,
		mobile_base.HeadClient.TILT_UP = -0.92,
		mobile_base.HeadClient.TILT_NEUTRAL = 0.0,
        mobile_base.HeadClient.TILT_DOWN = 0.29,
		mobile_base.HeadClient.EYES_OPEN = 0.0,
		mobile_base.HeadClient.EYES_NEUTRAL = 0.1,
		mobile_base.HeadClient.EYES_CLOSED = 0.41,
		mobile_base.HeadClient.EYES_HAPPY = -0.16,
		mobile_base.HeadClient.EYES_SUPER_SAD = 0.15,
		mobile_base.HeadClient.EYES_CLOSED_BLINK = 0.35
        """
    def __init__(self):

        self.behaviors = np.load('behaviors.npy')

        rospy.Subscriber('kuri_animation', UInt16, self.play_behavior)

        self._joint_states = mobile_base.JointStates()
        self._head_client = mobile_base.HeadClient(self._joint_states)
        assert self._head_client.wait_for_server(timeout=rospy.Duration(30.0))

        # At the start, open Kuri's eyes and point the head up:
        self._head_client.pan_and_tilt(
            pan=mobile_base.HeadClient.PAN_NEUTRAL,
            tilt=mobile_base.HeadClient.TILT_NEUTRAL,
            duration=1.0
        )

        self._head_client.eyes_to(
            radians=mobile_base.HeadClient.EYES_NEUTRAL,
            duration=0.5
        )

        
        #self.timer = rospy.Timer(rospy.Duration(0.1), self.move_head)
        self.state = Command()

    def play_behavior(self, msg):
        behavior_index = msg.data
        behavior = self.behaviors[behavior_index]

        for state in behavior:

            self._head_client.pan_and_tilt(
                    pan=state[0],
                    tilt=state[1],
                    duration=0.05
                )
            self._head_client.eyes_to(
                    radians=state[2],
                    duration=0.05
                )
            rospy.sleep(0.05)

    def shutdown(self):
        #self.timer.shutdown()
        print("Shutting down...")

if __name__ == '__main__':
    rospy.init_node('behavior_player')
    h = HeadTeleop()
    rospy.spin()
