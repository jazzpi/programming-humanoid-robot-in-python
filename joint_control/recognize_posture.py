'''In this exercise you need to use the learned classifier to recognize current posture of robot

* Tasks:
    1. load learned classifier in `PostureRecognitionAgent.__init__`
    2. recognize current posture in `PostureRecognitionAgent.recognize_posture`

* Hints:
    Let the robot execute different keyframes, and recognize these postures.

'''


import pickle
from angle_interpolation import AngleInterpolationAgent
from keyframes import hello
from os import listdir


class PostureRecognitionAgent(AngleInterpolationAgent):
    ROBOT_POSE_CLF = 'robot_pose.pkl'
    ROBOT_POSE_DIR = 'robot_pose_data'
    JOINTS = ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch',
              'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch']
    poses = None

    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(PostureRecognitionAgent, self).__init__(
            simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.poses = sorted(listdir(self.ROBOT_POSE_DIR))
        self.posture = 'unknown'
        with open(self.ROBOT_POSE_CLF) as fh:
            self.posture_classifier = pickle.load(fh)

    def think(self, perception):
        self.posture = self.recognize_posture(perception)
        return super(PostureRecognitionAgent, self).think(perception)

    def recognize_posture(self, perception):
        posture = 'unknown'
        # Set up data in the same format as in training
        data = map(lambda joint: perception.joint[joint], self.JOINTS)
        data += perception.imu[:2]

        pose = self.posture_classifier.predict([data])[0]
        return self.poses[pose]

        return posture


if __name__ == '__main__':
    agent = PostureRecognitionAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
