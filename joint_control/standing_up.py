'''In this exercise you need to put all code together to make the robot be able to stand up by its own.

* Task:
    complete the `StandingUpAgent.standing_up` function, e.g. call keyframe motion corresponds to current posture

'''


from recognize_posture import PostureRecognitionAgent
from keyframes import (hello, leftBackToStand, leftBellyToStand,
                       rightBackToStand, rightBellyToStand, wipe_forehead)


class StandingUpAgent(PostureRecognitionAgent):
    MOTION_FOR_POSE = {
        'Back': leftBackToStand,
        'Belly': leftBellyToStand,
    }

    def __init__(self, *args, **kwargs):
        super(StandingUpAgent, self).__init__(*args, **kwargs)
        self._end_of_motion = -1e-9
        self.stiffness_on_off_time = 0  # We need to check this in standing_up

    def think(self, perception):
        self.standing_up(perception)
        return super(StandingUpAgent, self).think(perception)

    def standing_up(self, perception):
        if perception.time - self.stiffness_on_off_time < \
                TestStandingUpAgent.STIFFNESS_OFF_CYCLE:
            # Joints are off, don't even bother
            print('Joints are off ({})'.format(
                perception.time - self.stiffness_on_off_time
            ))
            self.initial_time = None
            return

        motion = self.MOTION_FOR_POSE.get(self.posture, None)
        print(perception.time, self._end_of_motion)
        if motion is not None and perception.time > self._end_of_motion:
            self._current_motion = motion
            self.keyframes = motion()

            times = self.keyframes[1]
            max_time = 0
            for joint in times:
                joint_max = max(joint)
                if joint_max > max_time:
                    max_time = joint_max

            self._end_of_motion = perception.time + max_time

            # Force re-initialization of the data
            self.initial_time = None


class TestStandingUpAgent(StandingUpAgent):
    '''this agent turns off all motor to falls down in fixed cycles
    '''
    STIFFNESS_ON_CYCLE = 10  # in seconds
    STIFFNESS_OFF_CYCLE = 3  # in seconds

    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(TestStandingUpAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)

    def think(self, perception):
        action = super(TestStandingUpAgent, self).think(perception)
        time_now = perception.time
        if time_now - self.stiffness_on_off_time < self.STIFFNESS_OFF_CYCLE:
            action.stiffness = {j: 0 for j in self.joint_names}  # turn off joints
        else:
            action.stiffness = {j: 1 for j in self.joint_names}  # turn on joints
        if time_now - self.stiffness_on_off_time > self.STIFFNESS_ON_CYCLE + self.STIFFNESS_OFF_CYCLE:
            self.stiffness_on_off_time = time_now

        return action


if __name__ == '__main__':
    agent = TestStandingUpAgent()
    agent.run()
