#!/usr/bin/env python2

'''In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    the local_trans has to consider different joint axes and link parameters for different joints
'''

from math import sin, cos, sqrt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from numpy.matlib import matrix, identity

from angle_interpolation import AngleInterpolationAgent


class ForwardKinematicsAgent(AngleInterpolationAgent):
    # Joint distances from
    # http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html#head ff.
    DISTANCES = {
        # Head
        'HeadYaw': [0, 0, 126.5],
        'HeadPitch': [0, 0, 0],

        # Left arm
        'LShoulderPitch': [0, 98, 100],
        'LShoulderRoll': [0, 0, 0],
        'LElbowYaw': [105, 15, 0],
        'LElbowRoll': [0, 0, 0],

        # Right arm
        'RShoulderPitch': [0, -98, 100],
        'RShoulderRoll': [0, 0, 0],
        'RElbowYaw': [105, -15, 0],
        'RElbowRoll': [0, 0, 0],

        # Left leg
        'LHipYawPitch': [0, 50, -85],
        'LHipRoll': [0, 0, 0],
        'LHipPitch': [0, 0, 0],
        'LKneePitch': [0, 0, -100],
        'LAnklePitch': [0, 0, -102.9],
        'LAnkleRoll': [0, 0, 0],

        # Right leg
        'RHipYawPitch': [0, -50, -85],
        'RHipRoll': [0, 0, 0],
        'RHipPitch': [0, 0, 0],
        'RKneePitch': [0, 0, -100],
        'RAnklePitch': [0, 0, -102.9],
        'RAnkleRoll': [0, 0, 0],
    }

    # Joint rotation axes from
    # http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html#head-joints ff.
    AXES = {
        # Head
        'HeadYaw': 'z',
        'HeadPitch': 'y',

        # Left arm
        'LShoulderPitch': 'y',
        'LShoulderRoll': 'z',
        'LElbowYaw': 'x',
        'LElbowRoll': 'z',

        # Right arm
        'RShoulderPitch': 'y',
        'RShoulderRoll': 'z',
        'RElbowYaw': 'x',
        'RElbowRoll': 'z',

        # Left leg
        'LHipYawPitch': 'yz',
        'LHipRoll': 'x',
        'LHipPitch': 'y',
        'LKneePitch': 'y',
        'LAnklePitch': 'y',
        'LAnkleRoll': 'x',

        # Right leg
        'RHipYawPitch': 'yz',
        'RHipRoll': 'x',
        'RHipPitch': 'y',
        'RKneePitch': 'y',
        'RAnklePitch': 'y',
        'RAnkleRoll': 'x',
    }

    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(
            simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.transforms = {n: identity(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains = {
            'Head': ['HeadYaw', 'HeadPitch'],
            'LArm': ['LShoulderPitch', 'LShoulderRoll',
                     'LElbowYaw', 'LElbowRoll'],  # No wrist for us
            'RArm': ['RShoulderPitch', 'RShoulderRoll',
                     'RElbowYaw', 'RElbowRoll'],  # No wrist for us
            'LLeg': ['LHipYawPitch', 'LHipRoll', 'LHipPitch',
                     'LKneePitch', 'LAnklePitch', 'LAnkleRoll'],
            'RLeg': ['RHipYawPitch', 'RHipRoll', 'RHipPitch',
                     'RKneePitch', 'RAnklePitch', 'RAnkleRoll'],
        }

    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    def local_trans(self, joint_name, joint_angle):
        '''calculate local transformation of one joint

        :param str joint_name: the name of joint
        :param float joint_angle: the angle of joint in radians
        :return: transformation
        :rtype: 4x4 matrix
        '''
        T = identity(4)

        # Translation
        # `map` transforms row vector to column vector
        T[0:3, 3:4] = matrix(map(lambda x: [x], self.DISTANCES[joint_name]))

        # Rotation
        s = sin(joint_angle)
        c = cos(joint_angle)
        if self.AXES[joint_name] == 'x':
            T[0:3, 0:3] = matrix([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ])
        elif self.AXES[joint_name] == 'y':
            T[0:3, 0:3] = matrix([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ])
        elif self.AXES[joint_name] == 'z':
            T[0:3, 0:3] = matrix([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
        elif self.AXES[joint_name] == 'yz':
            # http://www.wolframalpha.com/input/?i=RotationMatrix%5Balpha,+%7B0,+1,+1%7D%5D
            s_sqrt_2 = s / sqrt(2)
            c_sub = 0.5 * (1 - c)
            c_add = 0.5 * (1 + c)
            T[0:3, 0:3] = matrix([
                [c, -s_sqrt_2, s_sqrt_2],
                [s_sqrt_2, c_add, c_sub],
                [-s_sqrt_2, c_sub, c_add]
            ])
        else:
            raise RuntimeError('Unknown axis {} (joint {})'.format(
                self.AXES[joint_name], joint_name))

        return T

    def forward_kinematics(self, joints):
        '''forward kinematics

        :param joints: {joint_name: joint_angle}
        '''
        for chain_joints in self.chains.values():
            T = identity(4)
            for joint in chain_joints:
                angle = joints[joint]
                Tl = self.local_trans(joint, angle)
                T *= Tl

                self.transforms[joint] = T.copy()
                T_ = T.round(2)


def plot(agent):
    fig = plt.figure()
    ax = Axes3D(fig)

    xs, ys, zs = [], [], []

    for chain_joints in agent.chains.values():
        for joint in chain_joints:
            vec = agent.transforms[joint] * [[1], [0], [0], [1]]
            T_ = agent.transforms[joint].round(2)
            xs.append(vec[0][0])
            ys.append(vec[1][0])
            zs.append(vec[2][0])

    ax.scatter(xs, ys, zs)

    # Set aspect ratio to 1:1:1
    # https://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio#comment40750814_19933125
    scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)

    plt.show()

if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    try:
        agent.run()
    except KeyboardInterrupt:
        plot(agent)
