#!/usr/bin/env python2

'''
In this exercise you need to know how to get sensor data.

* Task: get the current joint angle and temperature of joint HeadYaw

* Hint: The current sensor data of robot are store in perception (class
        Perception in spark_agent.py)

'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..',
                             'software_installation'))

from spark_agent import SparkAgent


class MyAgent(SparkAgent):
    def think(self, perception):
        angle = perception.joint['HeadYaw']
        temperature = perception.joint_temperature['HeadYaw']

        print 'HeadYaw angle: %f\t temperature: %f' % (angle, temperature)
        return super(MyAgent, self).think(perception)

if '__main__' == __name__:
    agent = MyAgent()
    agent.run()
