import numpy as np
from physics_sim import PhysicsSim

class MoveToTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, target_pos, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5.):
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
        self.action_repeat = 1

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.init_pose = self.sim.pose 
        self.old_pose = self.sim.pose

        # Goal
        self.target_pos = target_pos 

    def get_distance(self, a, b):
        return np.linalg.norm(a - b)

    """
    def get_reward(self, rotor_speeds):
        done = self.sim.next_timestep(rotor_speeds)
        if done:
            #Out of range, give a big punishment
            return (True, -1000.0)

        new_dist = self.get_distance(self.sim.pose[0:3], self.target_pos)
        old_dist = self.get_distance(self.old_pose[0:3], self.target_pos)        
        
        self.old_pose = self.sim.pose
        
        if new_dist < 0.1: 
            #Reach target, give a big reward
            return (True, 1000.0)
        else:
            #Reward each step on how much closer to the target
            return (False, old_dist - new_dist)
    """

    def get_reward(self, rotor_speeds):
        done = self.sim.next_timestep(rotor_speeds)
        if done :
            if self.sim.time < self.sim.runtime:
                #Out of scope
                print("crash")
                return (True, -1)
            else:            
                #Time out                   
                return (True, 1)
        
        r = 1 - abs(self.sim.pose[2] - self.target_pos[2]) / 300
        """Constraint movement to the z-axis and penlize based on how far a way from initial x and y"""
        r = r - min(1, self.get_distance(self.sim.pose[0:2], self.init_pose[0:2]) / 300)
        """reward for proposition of vertical velocity v[2]"""
        r = r + self.sim.v[2]/10 

        r = np.clip(r, -1, 1)

        return (False, r)



    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        all_reward = 0
        pose_all = []
        for i in range(self.action_repeat):
            done, reward = self.get_reward(rotor_speeds)
            all_reward += reward
            if done:
                for _ in range(i, self.action_repeat):
                    pose_all.append(self.sim.pose)
                break
            else:
                pose_all.append(self.sim.pose)
            
            print(rotor_speeds)
            print(self.sim.pose)
            print("reward %f"%(reward))


        next_state = np.concatenate(pose_all)

        return next_state, all_reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.old_pose = self.sim.pose
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
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
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state