'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''


import numpy as np
import sys
import json
from datetime import datetime

from pid import PIDAgent
from keyframes import (hello, leftBackToStand, leftBellyToStand,
                       rightBackToStand, rightBellyToStand, wipe_forehead)


class Recorder(object):
    '''Records joint angles'''
    def __init__(self):
        '''Create a new Recorder.'''
        self.record = {}
        self.handles = {}

    def append(self, data, time):
        '''Append new data points (for multiple joints).

        `data` should have the format
        `{'JointFoo': 0.123, 'JointBaz': 0.42, ...}`.'''
        for joint, val in data.iteritems():
            if joint not in self.record:
                self.record[joint] = {'x': [], 'y': []}
            self.record[joint]['x'].append(time)
            self.record[joint]['y'].append(val)

    def done(self, name, times, keys):
        '''Tell the recorder a given joint is done. `times` and `keys`
        should describe the Bezier handles.'''
        if name in self.handles:
            return
        self.handles[name] = (times, keys)

    def finish(self):
        '''Write data to disk.'''
        name = 'data-angle-interp-' + datetime.now().isoformat() + '.json'
        with open(name, 'w') as fh:
            json.dump({'record': self.record, 'handles': self.handles}, fh)


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(
            simspark_ip, simspark_port, teamname, player_id, sync_mode
        )
        self.initial_time = None
        self.initial_angles = {}
        self.recorder = Recorder()
        self.joints_done = {}
        self.keyframes = ([], [], [])

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def initialize_data(self, perception, names):
        self.initial_time = perception.time
        for joint in names:
            val = perception.joint.get(joint, None)
            if val is None:
                # print("WARN: joint {} is not in perception!".format(joint))
                val = 0
            self.initial_angles[joint] = val

    def angle_interpolation(self, keyframes, perception):
        target_joints = {}
        names, times, keys = keyframes
        if self.initial_time is None:
            self.initialize_data(perception, names)

        time = perception.time - self.initial_time

        if time <= 0:
            # Return initial for each joint
            target_joints = self.initial_angles
            self.recorder.append(target_joints, time)
            return target_joints

        for i in range(len(names)):
            joint_name = names[i]
            joint_times = times[i]
            joint_keys = keys[i]

            later = next((i for i, j in enumerate(joint_times) if j > time),
                         None)
            if later is None:
                target_joints[joint_name] = joint_keys[-1][0]
                if joint_name not in self.joints_done:
                    self.joints_done[joint_name] = True
                    self.recorder.done(joint_name, joint_times, joint_keys)
                    if len(self.joints_done) == len(names):
                        self.recorder.finish()
                continue

            points = self.bezier_points(time, joint_name, joint_times,
                                        joint_keys)
            if points is None:
                target_joints[joint_name] = joint_keys[-1][0]
                continue

            t = self.bezier_t(time, points)
            # prev_time = 0 if later == 0 else joint_times[later - 1]
            # next_time = joint_times[later]
            # t = (time - prev_time) / (next_time - prev_time)
            p0, p1, p2, p3 = points

            bezier = ((1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 +
                      3 * (1 - t) * t**2 * p2 + t**3 * p3)
            target_joints[joint_name] = bezier[1]

        self.recorder.append(target_joints, time)
        return target_joints

    def bezier_points(self, time, name, times, keys):
        '''Find the bezier points for a joint at a given time.'''
        later = next((i for i, j in enumerate(times) if j > time), None)
        if later is None:
            return None

        p0 = np.array([0, self.initial_angles[name]])
        p1 = np.array([0, self.initial_angles[name]])
        if later > 0:
            prev = keys[later - 1]
            p0 = np.array([times[later - 1], prev[0]])
            p1 = p0 + prev[2][1:]
        next_ = keys[later]
        p3 = np.array([times[later], next_[0]])
        p2 = p3 + next_[1][1:]

        return p0, p1, p2, p3

    @staticmethod
    def bezier_t(time, points):
        '''Find the t parameter of the bezier interpolation at a given time.'''
        t0, t1, t2, t3 = [p[0] for p in points]
        roots = np.roots([
            -1 * t0 + 3 * t1 - 3 * t2 + 1 * t3,  # i^3
             3 * t0 - 6 * t1 + 3 * t2,           # i^2
            -3 * t0 + 3 * t1,                    # i^1
             1 * t0 - time,                      # i^0
        ])
        # We only want the real solutions, but since it's solved numerically
        # they might have a small imaginary part
        r = set(
            map(lambda x: x.real,
                filter(lambda x: abs(x.imag) < 1e-9 and 0 <= x <= 1, roots)))
        if len(r) != 1:
            raise RuntimeError("{} time solutions at t={}".format(
                len(r), time))
        return r.pop()


if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    animations = [hello, leftBackToStand, leftBellyToStand, rightBackToStand,
                  rightBellyToStand, wipe_forehead]
    anim = 0
    if len(sys.argv) > 1:
        anim = int(sys.argv[1])
    agent.keyframes = animations[anim]()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
