import numpy
import os

qvel = os.listdir('/home/k38/Cassie_Ostrich_RL/test/ostrichrl/ostrichrl/assets/mocap/cassie/command_pos_vel/qpos')

i = 1
for file in qvel:
    file_data1 = numpy.load("/home/k38/Cassie_Ostrich_RL/test/ostrichrl/ostrichrl/assets/mocap/cassie/command_pos_vel/qvel/" + file)
    file_data = numpy.load("/home/k38/Cassie_Ostrich_RL/test/ostrichrl/ostrichrl/assets/mocap/cassie/qvel/" + file)
    print('cmd = ' + str(file_data1.shape) + ' qpos = ' + str(file_data.shape))
    i += 1

print(i)