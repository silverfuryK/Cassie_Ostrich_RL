import numpy
import os

qpos = os.listdir('./qpos')

# i = 1
# for file in qpos:
#     file_data = numpy.load("./qpos/" + file)
#     new_data = numpy.array([file_data[0], file_data[1], file_data[2]])
#     numpy.save("./qpos/" + str(i) + ".npy", new_data)
#     i += 1

# print(i)

i = 1
for file in qpos:
    file_data = numpy.load("./qvel/" + file)
    file2_data = numpy.load("./qpos/" + file)
    new_data = numpy.array([file_data[:,0], file_data[:,1], file_data[:,2]])
    new_data2 = numpy.array([file2_data[:,0], file2_data[:,1], file2_data[:,2]])
    numpy.save("./command_pos_vel/qvel/" + str(i) + ".npy", new_data)
    numpy.save("./command_pos_vel/qpos/" + str(i) + ".npy", new_data2)
    os.rename("./qpos/" + file, "./qpos/" + str(i) + ".npy")
    os.rename("./qvel/" + file, "./qvel/" + str(i) + ".npy")
    os.rename("./ximat/" + file, "./ximat/" + str(i) + ".npy")
    os.rename("./xipos/" + file, "./xipos/" + str(i) + ".npy")
    i += 1

print(i)