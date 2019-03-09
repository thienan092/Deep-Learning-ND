import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, action_repeat=1):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = action_repeat

        self.state_size = self.action_repeat * 6
        self.action_low = 1
        self.action_high = 900
        self.action_size = self.action_repeat * 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 10.
        if self.sim.pose[2] < (self.target_pos[2] - 3.): 
            reward += 2*(self.sim.pose[2] - self.sim.init_pose[2])
            reward -= (abs(self.sim.pose[:2] - self.target_pos[:2]) > 3.).sum()
        else: 
            reward += 10.
            reward -= 2*(abs(self.sim.pose[:3] - self.target_pos) > 3.).sum()
            reward -= np.int32(abs(self.sim.pose[2] - self.target_pos[2]) > 3.)
        
        if self.sim.done:
            if self.sim.time < self.sim.runtime: 
                reward -= 10
                
        reward = reward / 10
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds[(_*4):((_*4)+4)]) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, (reward / self.action_repeat), done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state