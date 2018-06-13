'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''

from numpy.matlib import identity, matrix
from numpy.linalg import det, norm
from scipy.optimize import fmin

from forward_kinematics import ForwardKinematicsAgent

class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        x0 = [0] * len(self.chains[effector_name])

        print('Starting optimization...')
        xopt = fmin(self._error_func, x0, args=(effector_name, transform))
        print('Done with optimization!')

        return dict(zip(self.chains[effector_name], xopt))

    @staticmethod
    def _last(gen):
        last = next(gen)
        for last in gen:
            pass
        return last

    def _error_func(self, thetas, effector, target):
        angles = dict(zip(self.chains[effector], thetas))
        T = self._last(self._forward_kinematics_for(effector, angles))
        rot_err = T * target.T  # Should be identity
        rot_err = abs(1 - abs(det(rot_err)))  # Determinant should thus be 1/-1
        trans_err = norm(T[0:3, 3:4] - target[0:3, 3:4])

        return rot_err * 100 + trans_err  # Rotation is more important than translation

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results'''
        angles = self.inverse_kinematics(effector_name, transform)
        targeted_joints = self.chains[effector_name]

        names, times, keys = [], [], []

        for joint in self.joint_names:
            names.append(joint)
            times.append([5.0, 8.0])

            joint_angles = (0.0, 0.0)
            if joint in targeted_joints:
                joint_angles = (0.0, angles[joint])
            keys.append([[joint_angles[0], [3, -1, 0], [3, 1, 0]],
                         [joint_angles[1], [3, -1, 0], [3, 1, 0]]])

        self.keyframes = (names, times, keys)  # the result joint angles have to fill in

if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    # Snapshot taken from hello animation
    T = matrix([[ 1.6000e-01, -2.7000e-01, -9.5000e-01,  1.8830e+01],
                [-6.7000e-01,  6.7000e-01, -3.1000e-01, -1.9234e+02],
                [ 7.2000e-01,  6.9000e-01, -7.0000e-02,  1.4467e+02],
                [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    agent.set_transforms('RArm', T)
    agent.run()
