from cassiemujoco import *
import time
import numpy as np
import math
import random

model = 'cassie.xml'
traj_path = 'ostrichrl/ostrichrl/assets/mocap/cassie/'
bot = CassieSim(model,terrain = False)
#vis = CassieVis(bot)
#u = pd_in_t()


class CassieEnv:
    def __init__(self, model, traj_path, simrate=240, clock_based=False):
        self.sim = CassieSim(model)
        self.vis = CassieVis(self.sim)
        self.traj_path = 'ostrichrl/ostrichrl/assets/mocap/cassie/'
        self.file_num = random.randint(1, 35)

        # NOTE: Xie et al uses full reference trajectory info
        # (i.e. clock_based=False)
        self.clock_based = clock_based

        if clock_based:
            self.observation_space = np.zeros(42)
            self.action_space      = np.zeros(10)
        else:
            self.observation_space = np.zeros(80)
            self.action_space      = np.zeros(10)
            
        self.qpos_targ = np.load(self.traj_path+"qpos/" + str(self.file_num) + ".npy")
        self.qvel_targ = np.load(self.traj_path+"qvel/" + str(self.file_num) + ".npy")
        self.cmd_vel_targ = np.load(self.traj_path+"command_pos_vel/qvel/" + str(self.file_num) + ".npy")
        self.xipos_targ = np.load(self.traj_path+"xipos/" + str(self.file_num) + ".npy")

        #print(self.qpos_targ.shape)
        #print(self.cmd_vel_targ.shape)
        #dirname = os.path.dirname(__file__)
        #if traj == "walking":
        #    traj_path = os.path.join(dirname, "trajectory", "stepdata.bin")

        #elif traj == "stand-in-place":
        #    raise NotImplementedError

        #self.trajectory = CassieTrajectory(traj_path)

        self.P = np.array([100,  100,  88,  96,  50]) 
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        self.u = pd_in_t()

        self.simrate = simrate # simulate X mujoco steps with same pd target
                               # 60 brings simulation from 2000Hz to roughly 30Hz

        self.time    = 0 # number of time steps in current episode
        self.phase   = 0 # portion of the phase the robot is in
        self.counter = 0 # number of phase cycles completed in episode

        ##### basic signals #######
        self.done = False

        # NOTE: a reference trajectory represents ONE phase cycle

        # should be floor(len(traj) / simrate) - 1
        # should be VERY cautious here because wrapping around trajectory
        # badly can cause assymetrical/bad gaits
        #self.phaselen = floor(len(self.trajectory) / self.simrate) - 1
        self.phaselen = math.floor(len(self.qpos_targ)/self.simrate) -1
        #print('PhaseLen = ' + str(self.phaselen),' | traj len = ' + str(len(self.qpos_targ)))

        # see include/cassiemujoco.h for meaning of these indices

        # // [ 6] Pelvis orientation qz
        # // [ 7] Left hip roll         (Motor [0])
        # // [ 8] Left hip yaw          (Motor [1])
        # // [ 9] Left hip pitch        (Motor [2])
        # // [10] Left achilles rod qw
        # // [11] Left achilles rod qx
        # // [12] Left achilles rod qy
        # // [13] Left achilles rod qz
        # // [14] Left knee             (Motor [3])
        # // [15] Left shin                        (Joint [0])
        # // [16] Left tarsus                      (Joint [1])
        # // [17] Left heel spring
        # // [18] Left foot crank
        # // [19] Left plantar rod
        # // [20] Left foot             (Motor [4], Joint [2])
        # // [21] Right hip roll        (Motor [5])
        # // [22] Right hip yaw         (Motor [6])
        # // [23] Right hip pitch       (Motor [7])
        # // [24] Right achilles rod qw
        # // [25] Right achilles rod qx
        # // [26] Right achilles rod qy
        # // [27] Right achilles rod qz
        # // [28] Right knee            (Motor [8])
        # // [29] Right shin                       (Joint [3])
        # // [30] Right tarsus                     (Joint [4])
        # // [31] Right heel spring

        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.pos_idx_eq = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
        #self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
        #self.pos_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        #self.pos_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])
        #self.vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        self.pos_index = np.array([7,8,9,14,15,16,20,21,22,23,28,29,30,34])
        self.vel_index = np.array([6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        self.ref_index = np.array([6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        # // [ 6] Left hip roll         (Motor [0])
        # // [ 7] Left hip yaw          (Motor [1])
        # // [ 8] Left hip pitch        (Motor [2])
        # // [12] Left knee             (Motor [3])
        # // [13] Left shin                        (Joint [0])
        # // [14] Left tarsus                      (Joint [1])
        # // [18] Left foot             (Motor [4], Joint [2])
        # // [19] Right hip roll        (Motor [5])
        # // [20] Right hip yaw         (Motor [6])
        # // [21] Right hip pitch       (Motor [7])
        # // [25] Right knee            (Motor [8])
        # // [26] Right shin                       (Joint [3])
        # // [27] Right tarsus                     (Joint [4])
        # // [31] Right foot            (Motor [9], Joint [5])
        
        self.get_init_pos()
        self.get_init_vel()
        self.get_init_geom()

        #init states to be impleented with the checkreset and reset function
        #self.init_qpos = 
        
    
    def step_simulation(self, phase):
        
        #add the target should be the actions added to the existing pos data
        
        #ref_pos, ref_vel = self.get_ref_state(self.phase + 1)

        #target = action + self.get_pos()
        ref_pose,_,_ = self.get_ref_state(phase)
        target = ref_pose[[0,1,2,3,6,7,8,9,10,13]]
        #target = target[[0,1,2,3,6,7,8,9,10,13]]
        #target = target[[ 7, 8, 9, 14, 20, 21, 22, 23, 28, 34]]

        self.u = pd_in_t()
        for i in range(5):
            self.u.leftLeg.motorPd.pGain[i]  = self.P[i]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i]

            self.u.leftLeg.motorPd.dGain[i]  = self.D[i]
            self.u.rightLeg.motorPd.dGain[i] = self.D[i]

            self.u.leftLeg.motorPd.torque[i]  = 0 # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0 

            self.u.leftLeg.motorPd.pTarget[i]  = target[i]
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        self.sim.step_pd(self.u)

    def step(self):
        self.action_t = [0,0,0,0,0,0,0,0,0,0]
        self.phase += 1
        for _ in range(self.simrate):
            self.step_simulation(self.phase)
            

        height = self.sim.qpos()[2]

        self.time  += 1
        

        #print(self.phase)

        # if self.phase > self.phaselen:
        #     self.phase = 0
        #     self.counter += 1
        self.counter += 1

        # Early termination
        #done = not(height > 0.4 and height < 3.0)

        

        reward = self.compute_reward()

        f = self.check_reset()

        #if f:
        #    reward = reward - 500


        # TODO: make 0.3 a variable/more transparent
        # if reward < 0.3:
        #     self.done = True

        #print(self.get_full_state().shape)
        #print(reward)
        #print(self.phase)

        return self.get_full_state(), reward, self.done, {}

    def reset(self):
        #self.phase = random.randint(0, self.phaselen)
        self.phase = 0
        self.time = 0
        self.counter = 0

        self.file_num = random.randint(1, 35)
        self.reward = 0

        self.qpos_targ = np.load(self.traj_path+"qpos/" + str(self.file_num) + ".npy")
        #print(self.qpos_targ.shape)
        self.qvel_targ = np.load(self.traj_path+"qvel/" + str(self.file_num) + ".npy")
        self.cmd_vel_targ = np.load(self.traj_path+"command_pos_vel/qvel/" + str(self.file_num) + ".npy")
        #print(self.cmd_vel_targ.shape)
        self.xipos_targ = np.load(self.traj_path+"xipos/" + str(self.file_num) + ".npy")

        self.phaselen = math.floor(len(self.qpos_targ)/self.simrate) -1
        print('PhaseLen = ' + str(self.phaselen),' | traj len = ' + str(len(self.qpos_targ)))

        self.action_t = np.zeros(10)
        #self.step(self.action_t)

        #qpos, qvel = self.get_ref_state(self.phase)

        self.sim.full_reset()
        
        

        ###### RESET THE POSE TO THE SAME AS THE POSE IN TRAJECTORY ##########

        #for i in range(5000):

        # self.sim.set_qpos(self.init_pose)
        # self.sim.set_qvel(self.init_vel)
        # self.sim.set_geom_pos(self.init_geom)
        self.sim.set_geom_pos([0,0,-10],'cassie-pelvis')
        self.sim.set_body_pos(name='cassie-pelvis', data=[0,0,2])
        
        
        #self.sim.step_pd(pd_in_t())
        #self.sim.set_geom_pos()
        #self.sim.set_qpos(self.get_init_pos())
        #self.sim.set_qvel(self.get_init_vel())
        #self.sim.set_geom_pos(self.get_init_geom())
        self.sim.hold()

        time.sleep(0.1)

        return self.get_full_state()

    # used for plotting against the reference trajectory
    def reset_for_test(self):
        self.phase = 0
        self.time = 0
        self.counter = 0

        qpos, qvel = self.get_ref_state(self.phase)

        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)

        time.sleep(1)

        return self.get_full_state()

    # check reset condition in step'
    def check_reset(self):
        #print(self.sim.xpos('cassie-pelvis')[2])
        if self.phase == len(self.qpos_targ)-1:
            self.done = True
            self.reset()
            return True
        
        if self.sim.xpos('cassie-pelvis')[2] < 0.4:
            #self.reward = self.reward - 500
            self.done = True
            self.reset()
            return True
        else:
            self.done = False
            return False

    
    def set_joint_pos(self, jpos, fbpos=None, iters=5000):
        """
        Kind of hackish. 
        This takes a floating base position and some joint positions
        and abuses the MuJoCo solver to get the constrained forward
        kinematics. 
        There might be a better way to do this, e.g. using mj_kinematics
        """

        # actuated joint indices
        joint_idx = [7, 8, 9, 14, 20,
                     21, 22, 23, 28, 34]

        # floating base indices
        fb_idx = [0, 1, 2, 3, 4, 5, 6]

        for _ in range(iters):
            qpos = np.copy(self.sim.qpos())
            qvel = np.copy(self.sim.qvel())

            qpos[joint_idx] = jpos

            if fbpos is not None:
                qpos[fb_idx] = fbpos

            self.sim.set_qpos(qpos)
            self.sim.set_qvel(0 * qvel)

            self.sim.step_pd(pd_in_t())


    # NOTE: this reward is slightly different from the one in Xie et al
    # see notes for details
    def compute_reward(self):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel())

        ref_pos, ref_vel, targ_vel = np.copy(self.get_ref_state(self.phase-1))

        weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]

        joint_error       = 0
        com_error         = 0
        orientation_error = 0
        spring_error      = 0
        vel_error         = 0

        # each joint pos
        # for i, j in enumerate(self.pos_idx):
        #     target = ref_pos[j]
        #     actual = qpos[j]

        joint_scale_weight = [0.8,0.8,0.8,1,1,0.8,0.8,0.8,1,1]


        #     joint_error += 30 * weight[i] * (target - actual) ** 2

        target = ref_pos[[0,1,2,3,6,7,8,9,10,13]]
        actual = np.array(qpos[self.pos_idx]) * np.array(joint_scale_weight)
        joint_error = sum(2 * (weight * (target-actual.tolist())) ** 2)
        

        # center of mass: x, y, z
        ## LOOK AT THE COM IN THE REF TRAJ AND THE INITIALIZED ONE!!!!
        #print("pos error")
        for j in [0, 1, 2]:
            target = ref_pos[j]
            actual = qpos[j]
            #print('targ = '+str(target)+'actual = '+str(actual))

            # NOTE: in Xie et al y target is 0

            com_error += 10 * (target - actual) ** 2
               
        # COM orientation: qx, qy, qz
        #print("orientation error")
        for j in [4, 5, 6]:
            target = ref_pos[j] # NOTE: in Xie et al orientation target is 0
            actual = qpos[j]
            #print('targ = '+str(target)+'actual = '+str(actual))

            orientation_error += (target - actual) ** 2
        #print("--") 
        
        # left and right shin springs
        #for i in [15, 29]:
        #    target = ref_pos[i] # NOTE: in Xie et al spring target is 0
        #    actual = qpos[i]

        #    spring_error += 1000 * (target - actual) ** 2     

        # target vel error

        for j in [0, 1 ,2]:
            target = targ_vel[j] # NOTE: in Xie et al orientation target is 0
            actual = qvel[j]
            #print('targ = '+str(target)+'actual = '+str(actual))

            vel_error += (target - actual) ** 2
        #print("--") 
        
        ### using exp ###
        # self.reward = 0.5 * np.exp(-joint_error) +       \
        #          0.3 * np.exp(-com_error) +         \
        #          0.0 * np.exp(-orientation_error) + \
        #          0.1 * np.exp(-spring_error) + \
        #          0.4 * np.exp(-vel_error)

        ### NOT using exp ###

        self.reward = 0.8 * np.exp(- 0.4* joint_error) +       \
                 0.0 * np.exp(-com_error) +         \
                 0.0 * np.exp(-orientation_error) + \
                 0.2 * np.exp(-spring_error) + \
                 0.0 * np.exp(-vel_error) + \
                    0

        return self.reward

    # get the corresponding state from the reference trajectory for the current phase
    def get_ref_state(self, phase=None):

        # if phase is None:
        #     phase = self.phase

        # if phase > self.phaselen:
        #     phase = 0

        #pos = np.copy(self.trajectory.qpos[phase * self.simrate])
        pos = np.copy(self.qpos_targ[phase,self.ref_index])
        #print(np.shape(pos))
        targ_vel = np.copy(self.cmd_vel_targ[:,phase])

        # this is just setting the x to where it "should" be given the number
        # of cycles
        #pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter
        
        # ^ should only matter for COM error calculation,
        # gets dropped out of state variable for input reasons

        # setting lateral distance target to 0
        #pos[1] = 0

        #vel = np.copy(self.trajectory.qvel[phase * self.simrate])
        vel = np.copy(self.qvel_targ[phase,self.ref_index])

        return pos, vel, targ_vel

    def get_full_state(self):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel()) 

        ref_pos, ref_vel, cmd_vel = self.get_ref_state(self.phase-1)
        #print(ref_pos)



        # this is everything except pelvis x and qw, achilles rod quaternions, 
        # and heel spring/foot crank/plantar rod angles
        # NOTE: x is forward dist, y is lateral dist, z is height

        # makes sense to always exclude x because it is in global coordinates and
        # irrelevant to phase-based control. Z is inherently invariant to (flat)
        # trajectories despite being global coord. Y is only invariant to straight
        # line trajectories.

        # [ 0] Pelvis y
        # [ 1] Pelvis z
        # [ 2] Pelvis orientation qw
        # [ 3] Pelvis orientation qx
        # [ 4] Pelvis orientation qy
        # [ 5] Pelvis orientation qz
        # [ 6] Left hip roll         (Motor [0])
        # [ 7] Left hip yaw          (Motor [1])
        # [ 8] Left hip pitch        (Motor [2])
        # [ 9] Left knee             (Motor [3])
        # [10] Left shin                        (Joint [0])
        # [11] Left tarsus                      (Joint [1])
        # [12] Left foot             (Motor [4], Joint [2])
        # [13] Right hip roll        (Motor [5])
        # [14] Right hip yaw         (Motor [6])
        # [15] Right hip pitch       (Motor [7])
        # [16] Right knee            (Motor [8])
        # [17] Right shin                       (Joint [3])
        # [18] Right tarsus                     (Joint [4])
        # [19] Right foot            (Motor [9], Joint [5])

        #pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34]) # 20 in nos
        #pos_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        #test config

        pos_index = np.array([7,8,9,14,15,16,20,21,22,23,28,29,30,34]) # 14 in nos
        # [ 0] Pelvis x
        # [ 1] Pelvis y
        # [ 2] Pelvis z
        # [ 3] Pelvis orientation wx
        # [ 4] Pelvis orientation wy
        # [ 5] Pelvis orientation wz
        # [ 6] Left hip roll         (Motor [0])
        # [ 7] Left hip yaw          (Motor [1])
        # [ 8] Left hip pitch        (Motor [2])
        # [ 9] Left knee             (Motor [3])
        # [10] Left shin                        (Joint [0])
        # [11] Left tarsus                      (Joint [1])
        # [12] Left foot             (Motor [4], Joint [2])
        # [13] Right hip roll        (Motor [5])
        # [14] Right hip yaw         (Motor [6])
        # [15] Right hip pitch       (Motor [7])
        # [16] Right knee            (Motor [8])
        # [17] Right shin                       (Joint [3])
        # [18] Right tarsus                     (Joint [4])
        # [19] Right foot            (Motor [9], Joint [5])
        
        #vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        vel_index = np.array([6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        if self.clock_based:
            #qpos[self.pos_idx] -= ref_pos[self.pos_idx]
            #qvel[self.vel_idx] -= ref_vel[self.vel_idx]

            clock = [np.sin(2 * np.pi *  self.phase / self.phaselen),
                     np.cos(2 * np.pi *  self.phase / self.phaselen)]
            
            ext_state = clock

        else:
            #ext_state = np.concatenate([ref_pos[pos_index], ref_vel[vel_index]])
            ext_state = np.concatenate([ref_pos, ref_vel])

        return np.concatenate([qpos[pos_index], 
                               qvel[vel_index], 
                               ext_state])
    def get_pos(self):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel()) 

        #ref_pos, ref_vel = self.get_ref_state(self.phase + 1)
        #print(ref_pos)

        # this is everything except pelvis x and qw, achilles rod quaternions, 
        # and heel spring/foot crank/plantar rod angles
        # NOTE: x is forward dist, y is lateral dist, z is height

        # makes sense to always exclude x because it is in global coordinates and
        # irrelevant to phase-based control. Z is inherently invariant to (flat)
        # trajectories despite being global coord. Y is only invariant to straight
        # line trajectories.

        # [ 0] Pelvis y
        # [ 1] Pelvis z
        # [ 2] Pelvis orientation qw
        # [ 3] Pelvis orientation qx
        # [ 4] Pelvis orientation qy
        # [ 5] Pelvis orientation qz
        # [ 6] Left hip roll         (Motor [0])
        # [ 7] Left hip yaw          (Motor [1])
        # [ 8] Left hip pitch        (Motor [2])
        # [ 9] Left knee             (Motor [3])
        # [10] Left shin                        (Joint [0])
        # [11] Left tarsus                      (Joint [1])
        # [12] Left foot             (Motor [4], Joint [2])
        # [13] Right hip roll        (Motor [5])
        # [14] Right hip yaw         (Motor [6])
        # [15] Right hip pitch       (Motor [7])
        # [16] Right knee            (Motor [8])
        # [17] Right shin                       (Joint [3])
        # [18] Right tarsus                     (Joint [4])
        # [19] Right foot            (Motor [9], Joint [5])
        
        #pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
        #pos_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        pos_index = np.array([7,8,9,14,15,16,20,21,22,23,28,29,30,34]) # 14 in nos



        # [ 0] Pelvis x
        # [ 1] Pelvis y
        # [ 2] Pelvis z
        # [ 3] Pelvis orientation wx
        # [ 4] Pelvis orientation wy
        # [ 5] Pelvis orientation wz
        # [ 6] Left hip roll         (Motor [0])
        # [ 7] Left hip yaw          (Motor [1])
        # [ 8] Left hip pitch        (Motor [2])
        # [ 9] Left knee             (Motor [3])
        # [10] Left shin                        (Joint [0])
        # [11] Left tarsus                      (Joint [1])
        # [12] Left foot             (Motor [4], Joint [2])
        # [13] Right hip roll        (Motor [5])
        # [14] Right hip yaw         (Motor [6])
        # [15] Right hip pitch       (Motor [7])
        # [16] Right knee            (Motor [8])
        # [17] Right shin                       (Joint [3])
        # [18] Right tarsus                     (Joint [4])
        # [19] Right foot            (Motor [9], Joint [5])
        
        #vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        vel_index = np.array([6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        #if self.clock_based:
            #qpos[self.pos_idx] -= ref_pos[self.pos_idx]
            #qvel[self.vel_idx] -= ref_vel[self.vel_idx]

        #    clock = [np.sin(2 * np.pi *  self.phase / self.phaselen),
        #             np.cos(2 * np.pi *  self.phase / self.phaselen)]
            
        #    ext_state = clock

        #else:
            #ext_state = np.concatenate([ref_pos[pos_index], ref_vel[vel_index]])
        #    ext_state = np.concatenate([ref_pos, ref_vel])

        return np.concatenate([qpos[pos_index]])
    
    
    def get_vel(self):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel()) 

        #ref_pos, ref_vel = self.get_ref_state(self.phase + 1)
        #print(ref_pos)

        # this is everything except pelvis x and qw, achilles rod quaternions, 
        # and heel spring/foot crank/plantar rod angles
        # NOTE: x is forward dist, y is lateral dist, z is height

        # makes sense to always exclude x because it is in global coordinates and
        # irrelevant to phase-based control. Z is inherently invariant to (flat)
        # trajectories despite being global coord. Y is only invariant to straight
        # line trajectories.

        # [ 0] Pelvis y
        # [ 1] Pelvis z
        # [ 2] Pelvis orientation qw
        # [ 3] Pelvis orientation qx
        # [ 4] Pelvis orientation qy
        # [ 5] Pelvis orientation qz
        # [ 6] Left hip roll         (Motor [0])
        # [ 7] Left hip yaw          (Motor [1])
        # [ 8] Left hip pitch        (Motor [2])
        # [ 9] Left knee             (Motor [3])
        # [10] Left shin                        (Joint [0])
        # [11] Left tarsus                      (Joint [1])
        # [12] Left foot             (Motor [4], Joint [2])
        # [13] Right hip roll        (Motor [5])
        # [14] Right hip yaw         (Motor [6])
        # [15] Right hip pitch       (Motor [7])
        # [16] Right knee            (Motor [8])
        # [17] Right shin                       (Joint [3])
        # [18] Right tarsus                     (Joint [4])
        # [19] Right foot            (Motor [9], Joint [5])
        
        #pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
        #pos_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        pos_index = np.array([7,8,9,14,15,16,20,21,22,23,28,29,30,34]) # 14 in nos

        # [ 0] Pelvis x
        # [ 1] Pelvis y
        # [ 2] Pelvis z
        # [ 3] Pelvis orientation wx
        # [ 4] Pelvis orientation wy
        # [ 5] Pelvis orientation wz
        # [ 6] Left hip roll         (Motor [0])
        # [ 7] Left hip yaw          (Motor [1])
        # [ 8] Left hip pitch        (Motor [2])
        # [ 9] Left knee             (Motor [3])
        # [10] Left shin                        (Joint [0])
        # [11] Left tarsus                      (Joint [1])
        # [12] Left foot             (Motor [4], Joint [2])
        # [13] Right hip roll        (Motor [5])
        # [14] Right hip yaw         (Motor [6])
        # [15] Right hip pitch       (Motor [7])
        # [16] Right knee            (Motor [8])
        # [17] Right shin                       (Joint [3])
        # [18] Right tarsus                     (Joint [4])
        # [19] Right foot            (Motor [9], Joint [5])
        
        #vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        vel_index = np.array([6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        #if self.clock_based:
            #qpos[self.pos_idx] -= ref_pos[self.pos_idx]
            #qvel[self.vel_idx] -= ref_vel[self.vel_idx]

        #    clock = [np.sin(2 * np.pi *  self.phase / self.phaselen),
        #            np.cos(2 * np.pi *  self.phase / self.phaselen)]
            
        #    ext_state = clock

        #else:
            #ext_state = np.concatenate([ref_pos[pos_index], ref_vel[vel_index]])
        #    ext_state = np.concatenate([ref_pos, ref_vel])

        return np.concatenate([qvel[vel_index]])

    def get_init_pos(self):
        #self.init_pose = np.copy(self.get_pos())
        self.init_pose = np.copy(self.sim.qpos())
        #print(self.init_pose)
        #print(self.qpos_targ[:,self.phase][0])
        for i, j in enumerate(self.pos_index):
            self.init_pose[j] = np.copy(self.qpos_targ[self.phase][i])

        return self.get_pos()

    def get_init_vel(self):
        #self.init_vel = np.copy(self.get_vel())
        self.init_vel = np.copy(self.sim.qvel())
        for i, j in enumerate(self.vel_index):
            self.init_vel[j] = np.copy(self.qvel_targ[self.phase][i])

        return self.get_vel()

#NEED TO WORK ON GEOM
    def get_init_geom(self):
        self.init_geom = np.copy(self.sim.get_geom_pos('cassie-pelvis'))
        #self.init_geom = np.copy(self.sim.xpos("cassie-pelvis"))
        #print(self.init_pose)
        #print(self.qpos_targ[:,self.phase][0])
        for i in range(3):
            self.init_geom[i] = np.copy(self.xipos_targ[self.phase,0,i])
        return self.init_geom


    def render(self):
        if self.vis is None:
            self.vis = CassieVis()

        self.vis.draw(self.sim)