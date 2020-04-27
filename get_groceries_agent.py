import numpy as np
import scipy as sp
from quaternion import *
from scipy.spatial.transform import Rotation as R

# # steps to run this in Jupyterlab
# action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION) # See rlbench/action_modes.py for other action modes
# env         = Environment(action_mode, '', ObservationConfig(), False)
# task        = env.get_task(PutGroceriesInCupboard) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
# pose_sensor = NoisyObjectPoseSensor(env)
# agent = StandardAgent(env, task, pose_sensor)

# utility functions
def skew(x):
	return np.array([[0, -x[2], x[1]],
					[x[2], 0, -x[0]],
					[-x[1], x[0], 0]])

def sample_normal_pose(pos_scale, rot_scale):
	'''
	Samples a 6D pose from a zero-mean isotropic normal distribution
	'''
	pos = np.random.normal(scale=pos_scale)
		
	eps = skew(np.random.normal(scale=rot_scale))
	R = sp.linalg.expm(eps)
	quat_wxyz = from_rotation_matrix(R)

	return pos, quat_wxyz

class NoisyObjectPoseSensor:

	def __init__(self, env):
		self._env = env

		self._pos_scale = [0.005] * 3
		self._rot_scale = [0.01] * 3

	def get_poses(self, noisy=False):
		objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
		obj_poses = {}

		for obj in objs:
			name = obj.get_name()
			pose = obj.get_pose()

			pos, quat_wxyz = sample_normal_pose(self._pos_scale, self._rot_scale)
			gt_quat_wxyz = quaternion(pose[6], pose[3], pose[4], pose[5])
			perturbed_quat_wxyz = quat_wxyz * gt_quat_wxyz
            
			if noisy:
				pose[:3] += pos
				pose[3:] = [perturbed_quat_wxyz.x, perturbed_quat_wxyz.y, perturbed_quat_wxyz.z, perturbed_quat_wxyz.w]

			obj_poses[name] = pose

		return obj_poses

class StandardAgent():

    def __init__(self,env,task,pose_sensor):
        self.env  = env
        self.task = task
        self.pose_sensor = pose_sensor
        self.gripped = False
        self._pos_scale = [0.1] * 3
        self._rot_scale = [0.1] * 3

    # AGENT METHODS
    def move_obj_to_safe_spot(self, obj):
        # obj: string 'soup'
        # identify safe spot for the object
        # grasp object
        # move the object
        # release the object
        
        print("going to grasp",obj)
        success = self.grasp(obj)

        # # waypoint3 is default pre-cupboard pose
        # waypoint = 'waypoint3'
        # waypoint_pose = self.pose_sensor.get_poses()[waypoint]
        # if success:
        #     print("going to pre-cupboard point")
        #     success = self.move_closer_to_goal(waypoint_pose[:3],waypoint_pose[3:],gripper=True,ignore_collisions=False)
        #     # success = self.move_to_pose(waypoint_pose[:3],waypoint_pose[3:],gripper=True,ignore_collisions=False)

        # the the grasped object to its safe spot
        waypoint = obj +'_safe_pose'
        waypoint_pose = self.pose_sensor.get_poses()[waypoint]
        if success:
            print("going to object safe spot")
            success = self.move_closer_to_goal(waypoint_pose[:3],waypoint_pose[3:],gripper=True,ignore_collisions=True)
            success = self.move_to_pose(waypoint_pose[:3],waypoint_pose[3:],gripper=True,ignore_collisions=True)
        
        # release
        if success: 
            print("releasing")
            success = self.move_to_pose(waypoint_pose[:3],waypoint_pose[3:],gripper=False,ignore_collisions=True)

    def move_obj_to_cupboard(self, obj):
        
        # grasp object first
        print("going to",obj,"grasp point")
        success = self.grasp(obj)

        # different waypoint for every object
        # waypoint3 is default pre-cupboard pose
        waypoint = 'waypoint3'
        waypoint_pose = self.pose_sensor.get_poses()[waypoint]
        if success:
            print("going to pre-cupboard point")
            # success = self.move_closer_to_goal(waypoint_pose[:3], waypoint_pose[3:], gripper=False, 
                                    # ignore_collisions=False)
            success = self.move_to_pose(waypoint_pose[:3],waypoint_pose[3:],gripper=True,ignore_collisions=True)

        # waypoint4 is default cupboard pos
        waypoint = 'waypoint4'
        waypoint_pose = self.pose_sensor.get_poses()[waypoint]
        if success:
            print("going to cupboard point")
            success = self.move_to_pose(waypoint_pose[:3],waypoint_pose[3:],gripper=True)

        # release
        if success: 
            print("releasing")
            success = self.move_to_pose(waypoint_pose[:3],waypoint_pose[3:],gripper=False)
        
        # move the gripper back outside the cupboard
        waypoint = 'waypoint3'
        waypoint_pose = self.pose_sensor.get_poses()[waypoint]
        if success:
            print("going to pre-cupboard point")
            success = self.move_to_pose(waypoint_pose[:3],waypoint_pose[3:],gripper=False,ignore_collisions=True)

        pass

    def get_push_point(self, gutter_pose, grasp_pose):
        """
        Input: gutter pose and grasp pose.
        Goal: Push from other direction in 10 cm distance.
        """
        dx = min(gutter_pose[0] - grasp_pose[0], 0)
        dy = gutter_pose[1] - grasp_pose[1]
        push_x = grasp_pose[0] - dx * (0.1/np.sqrt(dx**2 + dy**2)) 
        push_y = grasp_pose[1] - dy * (0.1/np.sqrt(dx**2 + dy**2))
        push_z = min(0.752 + 0.01, 0.752 + (grasp_pose[2] - 0.752) * 0.3)
    
        quat = [0, 1, 0, 0] # y rotation is 180 deg (for gripper to point down)
        z = np.arctan(dx / dy) + np.deg2rad(90.) # default gripper direction is perpendicular to pushing direction so rotate 90 degs
        z_rot = np.matrix([[np.cos(z), -np.sin(z), 0.],
                           [np.sin(z),  np.cos(z), 0.],
                           [0.       ,         0., 1.]])
    
        mtx = R.from_quat(quat).as_matrix() @ z_rot
        z_quat = R.from_matrix(mtx).as_quat()

        return [push_x, push_y, push_z], z_quat
    
    def get_possible_push_rotations(self, quat, offset):
        z_quats = np.zeros((2, 4),dtype=float)
        for i in range(2): #only try 2 orientations
            z = np.deg2rad(180*i + offset)
            z_rot = np.matrix([[np.cos(z), -np.sin(z), 0.],
                               [np.sin(z),  np.cos(z), 0.],
                               [0.       ,         0., 1.]])
            mtx = R.from_quat(quat).as_matrix() @ z_rot
            z_quats[i] = R.from_matrix(mtx).as_quat()

        return z_quats
    
    def get_valid_push_path(self, pos,quat,gripper,ignore_collisions=False,offset=0):
        success, path = self.find_path(pos, quat, gripper, ignore_collisions)
        max_itr = 1
        itr = 0
        while not success and itr<max_itr:
            itr += 1
            offset = np.random.randint(-10, 10) #random offset is much smaller +- 10 degrees

            print("-------- Trying with offset: ",offset)
            test_quats = self.get_possible_push_rotations(quat, offset) #0, 180

            i=1
            path = None
            for q in test_quats:
                success, path = self.find_path(pos, q, gripper, ignore_collisions)
                if success:
                    print('path found to quat', i)
                    break
                else:
                    print('path not found to quat', i)
                    i+=1
        
        if not success:
            print('failed')
        
        return success, path

    def push_obj_to_ledge(self, obj):
        gutter_pose = self.pose_sensor.get_poses()['gutter_pose']
        grasp_point = self.pose_sensor.get_poses()[obj+'_grasp_point']

        push_pos = self.get_push_point(gutter_pose, grasp_point)

        ############ make sure end effector is upright
        # get to push pose
        success = self.move_to_pose(push_pos, grasp_point[3:], gripper=False, 
                                    ignore_collisions=ig_col_during_move_to_grasp)        

        # push to gutter
        #success = self.move_to_pose(....)

        pass

    def grasp(self, obj):
        
        grasp_point = self.pose_sensor.get_poses()[obj + '_grasp_point']

        if obj == 'soup':
            offset  = 0.03
            ig_col_during_move_to_grasp = False
            custom_offset_grasp_point = self.get_new_offset_EE_position(grasp_point[:3], grasp_point[3:], offset)
        if obj == 'coffee':
            offset  = -0.05
            custom_offset_grasp_point = grasp_point
            grasp_point = self.get_new_offset_EE_position(grasp_point[:3], grasp_point[3:], offset)
            ig_col_during_move_to_grasp = True
        if obj == 'sugar':
            offset  = 0.00
            ig_col_during_move_to_grasp = False  
            custom_offset_grasp_point = self.get_new_offset_EE_position(grasp_point[:3], grasp_point[3:], offset)
        
        # grasp step (first move closer to the goal)
        # success = self.move_closer_to_goal(grasp_point[:3], grasp_point[3:], gripper=False, 
        #                             ignore_collisions=False)
        success = self.move_to_pose(grasp_point[:3], grasp_point[3:], gripper=False, 
                                    ignore_collisions=ig_col_during_move_to_grasp)
        print(obj + ' Pre-grasp done.')
        
        # final grasp step
        if success:
            success = self.move_to_pose(custom_offset_grasp_point[:3], custom_offset_grasp_point[3:],
                                        gripper=False, ignore_collisions=True)
            print(obj + 'grasp location done.')
        
        # gripper close step
        if success:
            success = self.move_to_pose(custom_offset_grasp_point[:3], custom_offset_grasp_point[3:],
                                        gripper=True, ignore_collisions=True)
            print(obj + 'grasped!')
        
        return success
    
    def move_to_pose(self,pos,quat,gripper,ignore_collisions=False):
        # Given a position and orientation of the gripper, the agent moves the gripper to that pose
        # Inputs:
            # pos: 3x1 vector of x,y,z location in world frame
            # quat : 4x1 vector of quaternion in world frame
            # gripper : boolean (T = CLOSED, F = OPEN)
        # Outputs: 
            # success: a boolean to represent path success
        # grpr_pose = self.env._robot.gripper.get_pose()

        # then go to the final pose
        success, path = self.get_valid_path(pos,quat,gripper,ignore_collisions)
        if success:
            success = self.execute_path(path)

        return success
    
    def move_to_push_pose(self,pos,quat,gripper,ignore_collisions=False):
        # Given a position and orientation of the gripper, the agent moves the gripper to that pose
        # Inputs:
            # pos: 3x1 vector of x,y,z location in world frame
            # quat : 4x1 vector of quaternion in world frame
            # gripper : boolean (T = CLOSED, F = OPEN)
        # Outputs: 
            # success: a boolean to represent path success
        # grpr_pose = self.env._robot.gripper.get_pose()

        # then go to the final pose
        success, path = self.get_valid_push_path(pos,quat,gripper,ignore_collisions)
        if success:
            success = self.execute_path(path)

        return success

    # HELPER METHODS ONLY USED BY THE AGENT------------------------------

    def get_close_enough_pose(self, pos, quat):
        
        position, quat_wxyz = sample_normal_pose(self._pos_scale, self._rot_scale)
        gt_quat_wxyz = quaternion(quat[3], quat[0], quat[1], quat[2])
        perturbed_quat_wxyz = quat_wxyz * gt_quat_wxyz
            
        pos += position
        quat = [perturbed_quat_wxyz.x, perturbed_quat_wxyz.y, perturbed_quat_wxyz.z, perturbed_quat_wxyz.w]
        
        return pos, quat

    def move_closer_to_goal(self,pos,quat,gripper,ignore_collisions=False):        
        
        pre_pos = np.copy(pos)
        pre_pos[2] += 0.10
        pre_quat = np.array([1,0,0,0])

        return self.move_to_pose(pre_pos, pre_quat, gripper, ignore_collisions)

    def get_valid_path(self,pos,quat,gripper,ignore_collisions=False,offset=0):
        # Try to get to the specified pos  quat,and quaternion by checking 4 orientations of quaternion
        # INPUTS
            # pos  quat,: x,y,z of waypoint
            # quat : quat of waypoint
            # gripper : T = closed, F = open
            # ignore_collisions : T = ignore, F = don't ignore
            # offset : degrees to offset before trying 4 quat poses (not used yet...)
        # OUTPUTS :
            # success : bool
        success, path = self.find_path(pos, quat, gripper, ignore_collisions)

        # if not success:
        max_itr = 1
        itr = 0
        while not success and itr<max_itr:
            itr += 1
            offset = np.random.randint(10,80)

            pos, quat = self.get_close_enough_pose(pos, quat)

            print("-------- Trying with offset: ",offset)
            test_quats = self.get_possible_rotations(quat,offset)
            i=1
            path = None
            for q in test_quats:
                success, path = self.find_path(pos, q, gripper, ignore_collisions)
                if success:
                    print('path found to quat', i)
                    break
                else:
                    print('path not found to quat', i)
                    i+=1
        
        if not success:
            print('failed')
        
        return success, path

    def find_path(self, pos, quat, gripper, ignore_collisions=False):
        # Tries to plan a path to the desired pos  quat,and quat
        # INPUTS
            # pos: 3x1 vector of x,y,z location in world frame
            # quat : 4x1 vector of quaternion in world frame
            # gripper : boolean (T = CLOSED, F = OPEN)
            # ignore_collisions (T = ignore, F = don't ignore)
        # OUTPUTS
            # success : boolean (T = path found, F = path not found)
            # path : array of steps to take (None if success == F)

        try:
            path_class = self.env._robot.arm.get_path(position=pos,quaternion=quat,ignore_collisions=ignore_collisions)
            path = path_class._path_points.reshape(-1,path_class._num_joints)
            if gripper:
                    gripper_poses = np.zeros((path.shape[0],1))
            else:
                gripper_poses = np.ones((path.shape[0],1))
            path = np.hstack((path, gripper_poses))
            success = True
        except:
            success = False
            path = None 

        return success, path

    def execute_path(self,path):
        # Executes a path
        # INPUT
            # path : array of joint positions returned by find_path
            # path generated is either 50 steps (linear) or 300 steps (nonlinear)
        for pose in path:
            obs, reward, terminate = self.task.step(pose)

        return True

    def get_possible_rotations(self,quat,offset=0):
        # Get rotations about z in 90 degree increments
        # INPUT
            # quaternion : waypoint quaternion
            # offset : how much off of 0 do we want to start doing the 90 degree transformations (for use if first try of this fxn returns no good paths)
        # OUTPUT
            # quats : 4 quaternions representing 0, 90, 180, 270 degree offset in Z from input quaternion
        quats = np.zeros((4,4))
        quats[0,:] = quat

        matrix = R.from_quat(quat).as_matrix()
        ninety = R.from_rotvec((np.pi/2 + offset) * np.array([1, 0, 0])).as_matrix()

        for i in range(1,4):
            matrix = np.dot(matrix, ninety)
            quats[i] = R.from_matrix(matrix).as_quat()

        return quats
    
    def test(self, obj, obs):
        starting_pose = obs.gripper_pose
        p0 = starting_pose[:3]
        q0 = starting_pose[3:]
        grasp_point = self.pose_sensor.get_poses()[obj + '_grasp_point']
        pos = grasp_point[:3]
        quat = grasp_point[3:]
        print(pos)
        print(quat)
        quats = self.get_possible_rotations(quat)
        i = 0
        for q in quats:
            print("trying",i)
            success,path = self.find_path(pos,q,False,True)
            if success:
                print("succeeded")
                self.execute_path(path)
                success,path = self.find_path(p0,q0,False,True)
                self.execute_path(path)
            else:
                print("failed")
            i+=90

    def quat_to_homo_trans_mat(self, quat_world):
        A = np.eye(4)
        A[:3,:3] = R.from_quat(quat_world).as_matrix()
        return A

    def pose_to_homo_trans_mat(self, pose):
        pos = pose[:3]
        quat = pose[3:]
        A = np.eye(4)
        A[:3,:3] = R.from_quat(quat).as_matrix()
        A[:,3] = [pos[0], pos[1], pos[2], 1]
        return A

    def get_new_offset_EE_position(self,position, quat, offset_amount):
    
        # Move the EE along the z-xis by the specified offset_amount
        # Inputs:
            # pos  quat,: x,y,z of waypoint in world frame
            # quat : quat of waypoint in world frame
            # offset_amount : offset_amount in z
        # Output:
            # new_position : x,y,z of waypoint in world frame

        # create rotation matrix between world frame and can frame
        world2gripper        = self.quat_to_homo_trans_mat(quat)
        world2gripper[0:3,3] = position

        # create translation vector in gripper frame
        translation_vec        = np.array([0, 0, offset_amount, 0]).reshape(-1, 1)
        translation_in_global  = world2gripper @ translation_vec
        new_position           = translation_in_global.flatten()[:3] + position

        return np.append(new_position, quat)