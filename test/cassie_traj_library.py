import numpy as np
import random

class CassieTrajectory:
    def __init__(self, filepath): # points to the cassie mocap folder
        self.traj_path = filepath
    
    def get_data(self, file_num =0):
        self.qpos_targ = np.load(self.traj_path+"qpos/" + str(self.file_num) + ".npy")
        #print(self.qpos_targ.shape)
        self.qvel_targ = np.load(self.traj_path+"qvel/" + str(self.file_num) + ".npy")
        self.cmd_vel_targ = np.load(self.traj_path+"command_pos_vel/qvel/" + str(self.file_num) + ".npy")
        self.xipos_targ = np.load(self.traj_path+"xipos/" + str(self.file_num) + ".npy")
        
        return self.qpos_targ, self.qvel_targ, self.cmd_vel_targ, self.xipos_targ
