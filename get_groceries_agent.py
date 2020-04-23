import numpy as np
import scipy as sp



class StandardAgent():

    def __init__(self):
        pass

    # tries to plan a path to the desired pose and quat
    # INPUTS
    # pose : 3x1 vector of x,y,z location in world frame
    # quat : 4x1 vector of quaternion in world frame
    # gripper : boolean (T = CLOSED, F = OPEN)
    # ignore_collisions (T = ignore, F = don't ignore)
    # OUTPUTS
    # success : boolean (T = path found, F = path not found)
    # path : array of steps to take (None if success == F)
    def find_path(self,pose,quat,gripper,ignore_collisions):
    	try:
			path_class = env._robot.arm.get_path(position=pose,quaternion=quat,ignore_collisions=ignore_collisions)
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

	# executes a path
	# INPUT
	# path : array of joint positions returned by find_path
	# path generated is either 50 steps (linear) or 300 steps (nonlinear)
	def execute_path(self,path):
		for pose in path:
			obs, reward, terminate = task.step(pose)

	# get rotations about z in 90 degree increments
	# INPUT
	# quaternion : waypoint quaternion
	# offset : how much off of 0 do we want to start doing the 90 degree transformations (for use if first try of this fxn returns no good paths)
	# OUTPUT
	# quats : 4 quaternions representing 0, 90, 180, 270 degree offset in Z from input quaternion
	def get_possible_rotations(self,quaternion,offset=0):
		quats = np.zeros((4,4))
		quats[0] = quaternion

		matrix = R.from_quat(quaternion).as_matrix()
		ninety = R.from_rotvec(np.pi/2 + offset * np.array([0, 0, 1])).as_matrix()

		for i in range(1,4):
			matrix = np.dot(ninety,matrix)
			quats[i] = R.from_matrix(matrix).as_quat()

		return quats

	# try to get to the specified pose and quaternion by checking 4 orientations of quaternion
	# INPUTS
	# pose : x,y,z of waypoint
	# quat : quat of waypoint
	# gripper : T = closed, F = open
	# ignore_collisions : T = ignore, F = don't ignore
	# offset : degrees to offset before trying 4 quat poses (not used yet...)
	# OUTPUTS :
	# success : bool
	def try_path(self,pose,quat,gripper,ignore_collisions,offset=0):
		test_quats = self.get_possible_rotations(quat,offset=0)
		i=1

		for q in test_quats:
			success, path = self.find_path(pose,q,gripper,ignore_collisions)
			if success:
				self.execute_path(path)
				print('path found to quat', i)
				break
			else:
				print('path not found to quat', i)
			i+=1

		return success

	def quat_to_homo_trans_mat(self,quat_world):
	    A = np.eye(4)
	    A[:3,:3] = R.from_quat(quat_world).as_matrix()
	    return A

	def find_can_pose(self,pose_world, quat_world, amount):
	    print("where i am", pose_world)
	    move_amount_along_z =  amount
	    # create rotation matrix between world frame and can frame
	    world2gripper        = quat_to_homo_trans_mat(quat_world)
	    world2gripper[0:3,3] = pose_world
	    # create translation vector in gripper frame
	    translation_vec      = np.array([0, 0, move_amount_along_z, 0]).reshape(-1, 1)
	    translation_in_global  = world2gripper @ translation_vec
	    newpose_world        = translation_in_global.flatten()[:3] + pose_world
	    print("where i wanna go",newpose_world)
	    return newpose_world

