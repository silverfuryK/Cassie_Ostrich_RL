import numpy as np

traj_path = 'ostrichrl/ostrichrl/assets/mocap/cassie/'
qpos = np.load(traj_path+"qpos/1.npy")

print(qpos.shape)
print(qpos[0,:].shape)