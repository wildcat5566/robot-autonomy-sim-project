import numpy as np
import scipy as sp
from quaternion import *

import sys

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
from scipy.spatial.transform import Rotation as R

###############################################################
# Functions given in baseline code

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


class RandomAgent:

	def act(self, obs):
		delta_pos = [(np.random.rand() * 2 - 1) * 0.005, 0, 0]
		delta_quat = [0, 0, 0, 1] # xyzw
		gripper_pos = [np.random.rand() > 0.5]
		return delta_pos + delta_quat + gripper_pos

	def act_abs(self,des):
		gripper_pos = [1]
		print("des len",type(des))
		print(len(des+gripper_pos))
 
		return list(des) + gripper_pos


class NoisyObjectPoseSensor:

	def __init__(self, env):
		self._env = env

		self._pos_scale = [0.005] * 3
		self._rot_scale = [0.01] * 3

	def get_poses(self):
		objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
		obj_poses = {}

		for obj in objs:
			name = obj.get_name()
			pose = obj.get_pose()

			pos, quat_wxyz = sample_normal_pose(self._pos_scale, self._rot_scale)
			gt_quat_wxyz = quaternion(pose[6], pose[3], pose[4], pose[5])
			perturbed_quat_wxyz = quat_wxyz * gt_quat_wxyz

			pose[:3] += pos
			pose[3:] = [perturbed_quat_wxyz.x, perturbed_quat_wxyz.y, perturbed_quat_wxyz.z, perturbed_quat_wxyz.w]

			obj_poses[name] = pose

		return obj_poses

###################################################################################################3
# Kevin's custom fxn's


def go_to_pose(position,quaternion,gripper,collisions):
	try:
		# ,algorithm=Algos.BiTRRT
		path_class = env._robot.arm.get_path(position=position, quaternion=quaternion,ignore_collisions=collisions)
		path = path_class._path_points.reshape(-1, path_class._num_joints)
		if gripper:
			gripper_poses = np.zeros((path.shape[0],1))
		else:
			gripper_poses = np.ones((path.shape[0],1))
		path = np.hstack((path, gripper_poses))
		path_class.visualize()
		for pose in path:
			obs, reward, terminate = task.step(pose)
		print("path found!")
		print(path.shape[0],"steps")
		return True
	except:
		print("path not found")
		return False

def find_can_pose(pose_world,quat_world):
	print("where i am", pose_world)

	# create rotation matrix between world frame and can frame
	rot = R.from_quat(quat_world).as_matrix()
	rot = np.hstack((rot,np.zeros((3,1))))
	rot = np.vstack((rot,np.zeros((1,4))))
	rot[-1,-1] = 1

	# create translation matrix between world frame and can frame
	trans = np.eye(4)
	trans[:3,-1] = -pose_world

	# combine rotation and translation to form transformation from world frame to gripper frame
	world2gripper = R.from_rotvec(np.pi * np.array([-1,0,-1])).as_matrix() # starting joint rot in global coordinates is -180 in x and -180 in z
	world2gripper = np.hstack((world2gripper,np.zeros((3,1))))
	world2gripper = np.vstack((world2gripper,np.zeros((1,4))))
	world2gripper[-1,-1] = 1
	world2gripper = np.dot(rot,trans)

	# offest the pose by 1cm, basically move towards the can a lil bit for a better grasp
	offset = np.eye(4)
	offset[2,-1] = -0.01
	world2gripper = np.dot(offset,world2gripper)

	# get pose of gripper in the gripper frame
	pose_gripper = np.dot(world2gripper,np.hstack((pose_world,np.ones(1))))
	print("pose in gripper frame",pose_gripper)

	# invert the transformation matrix to go back to world coordinates.
	# now the offset should be reflected in the new_pose_in
	gripper2world = np.eye(4)
	gripper2world[:3,:3] = world2gripper[:3,:3].T
	gripper2world[:3,3] = -np.dot(world2gripper[:3,:3],world2gripper[:3,-1])
	pose_gripper = np.zeros(4)
	pose_gripper[-1] = 1
	newpose_world = np.dot(gripper2world,pose_gripper)[:-1]
	# print(np.dot(gripper2world,pose_gripper))
	print("where i wanna go",newpose_world)

	return newpose_world

# get rotations about z in 90 degree increments
def get_possible_rotations(quaternion):
	quats = np.zeros((4,4))
	quats[0] = quaternion

	matrix = R.from_quat(quaternion).as_matrix()
	ninety = R.from_rotvec(np.pi/2 * np.array([0, 0, 1])).as_matrix()

	for i in range(1,4):
		matrix = np.dot(ninety,matrix)
		quats[i] = R.from_matrix(matrix).as_quat()

	return quats

# find a path
# returns a boolean for success and the path
def find_path(pose,quat,gripper,ignore_collisions):
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

	return success,path

def get_push_point(gutter, destination):
	dx = min(gutter[0] - destination[0], 0)
	dy = gutter[1] - destination[1]
	push_x = destination[0] - dx * (0.1/np.sqrt(dx**2 + dy**2)) 
	push_y = destination[1] - dy * (0.1/np.sqrt(dx**2 + dy**2))
	push_z = min(0.752 + 0.05, 0.752 + (destination[2] - 0.752) * 0.3)
	return push_x, push_y, push_z

# executes a path
def execute_path(path):
	for pose in path:
		obs, reward, terminate = task.step(pose)

if __name__ == "__main__":
	action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION) # See rlbench/action_modes.py for other action modes
	env = Environment(action_mode, '', ObservationConfig(), False)
	task = env.get_task(PutGroceriesInCupboard) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
	agent = RandomAgent()
	obj_pose_sensor = NoisyObjectPoseSensor(env)

	state = 'go_to_waypoint'
	grasppoints = ['soup_grasp_point', 'sugar_grasp_point', 'coffee_grasp_point']
	safeposes = ['soup_safe_pose', 'sugar_safe_pose', 'coffee_safe_pose']
	waypoint = 'coffee_safe_pose'

	descriptions, obs = task.reset()
	starting_pose = obs.gripper_pose
	pos0 = starting_pose[:3]
	quat0 = starting_pose[3:]

	print(R.from_quat(quat0).as_euler('xyz', degrees=True))
	print(pos0)

	obj_poses = obj_pose_sensor.get_poses()
	pre_cup_quat = obj_poses['waypoint3'][3:7]
	vec = R.from_quat(pre_cup_quat).as_rotvec()

	print()
	num_iterations = 0
	obj_id = 0
	push_z = None

	while True:
		print('****************')
		print("Current State:", state)
		if state == 'shutdown':
			print('shutting down...')
			print('****************')
			env.shutdown()
			break

		if state == 'push_object':
			obj_poses = obj_pose_sensor.get_poses()
			#print(obj_poses['gutter_pose'][:2])
			#print(obj_poses['soup_grasp_point'][:2])
			print("Waypoint:", waypoint)
			

			if waypoint == 'soup_grasp_point':
				#move to waypoint
				destination = obj_poses[waypoint]
				gutter = obj_poses['gutter_pose']
				
				push_x, push_y, push_z = get_push_point(gutter, destination) #0.05 for soup
				#print(push_z, destination[2])
				des_pose = [push_x, push_y, push_z]
				des_quat = destination[3:7] 

				quats = get_possible_rotations(des_quat)
				i = 1

				for quat in quats:
					print(np.rad2deg(R.from_quat(quat).as_rotvec()))
					success,path = find_path(des_pose,quat,gripper=True,ignore_collisions=False)

					if success:
						print("quat",i,"succeeded")
						execute_path(path)
						waypoint = 'gutter_pose' #wp3
						des_quat  = quat
						break

					else:
						print("quat",i,"failed")
					i+=1

				if not success:
					print("path to",waypoint,"not found")
					state = 'shutdown'
				else: 
					success,path = find_path(des_pose,quat,gripper=True,ignore_collisions=True)
					if success:
						execute_path(path)
						waypoint = 'gutter_pose' #wp3
					else: 
						print("push approach path not found")
						state = 'shutdown'

			elif waypoint == 'gutter_pose':
				destination = obj_poses['gutter_pose']
				des_pose = destination[:3]
				des_pose[2] = push_z
				#des_quat = destination[3:7]
				success,path = find_path(des_pose,des_quat,gripper=True,ignore_collisions=True)
				if success:
					execute_path(path)
					waypoint = 'waypoint2'
				else:
					print('path not found')
					state = 'shutdown'
				

			elif waypoint == 'waypoint2': #midpoint
				destination = obj_poses['waypoint2']
				des_pose = destination[:3]
				des_quat = destination[3:7]
				success,path = success,path = find_path(des_pose,des_quat,gripper=True,ignore_collisions=True)
				if success:
					execute_path(path)
					state = 'go_to_waypoint'
					waypoint = 'soup_grasp_point'
				else:
					print('path not found')
					state = 'shutdown'
				state = 'shutdown'
				

		elif state == 'go_to_waypoint':
			obj_poses = obj_pose_sensor.get_poses()
			print("Object id: ", str(obj_id))
			print("Waypoint:", waypoint)
			if waypoint == 'resetpoint':
				destination = obj_poses['waypoint2']
			else:
				destination = obj_poses[waypoint]
			des_pose = destination[:3]
			des_quat = destination[3:7]


			if waypoint == grasppoints[obj_id] or 'coffee_safe_pose':

				quats = get_possible_rotations(des_quat)
				i = 1

				for quat in quats:
					print(np.rad2deg(R.from_quat(quat).as_rotvec()))
					success,path = find_path(des_pose,quat,gripper=False,ignore_collisions=False)

					if success:
						print("quat",i,"succeeded")
						execute_path(path)
						waypoint = 'waypoint2' #2
						des_quat  = quat
						break

					else:
						print("quat",i,"failed")
					i+=1
				if not success:
					print("path to",waypoint,"not found")
					state = 'shutdown'
				"""else: 
					offset_can_pose = find_can_pose(des_pose,des_quat)
					success,path = find_path(offset_can_pose,des_quat,gripper=False,ignore_collisions=True)
					if success:
						execute_path(path)
						waypoint = 'waypoint2' #2
					else: 
						print("can approach path not found")
						state = 'shutdown'"""
				state = 'shutdown'

			"""elif waypoint == 'waypoint3': # 2 midpoint
				success,path = success,path = find_path(des_pose,des_quat,gripper=True,ignore_collisions=True)
				if success:
					execute_path(path)
					#waypoint = safeposes[obj_id]
					waypoint = 'coffee_cupboard_pose'
				else:
					print('path not found')
					state = 'shutdown'

			elif waypoint == 'coffee_cupboard_pose': # 2 midpoint
				success,path = success,path = find_path(des_pose,des_quat,gripper=True,ignore_collisions=True)
				if success:
					execute_path(path)
					waypoint = 'waypoint3'
				else:
					print('path not found')
					state = 'shutdown'
				

			elif waypoint == safeposes[obj_id]:
				success,path = success,path = find_path(des_pose,des_quat,gripper=True,ignore_collisions=False)
				if success:
					execute_path(path)
					waypoint = 'resetpoint'
				else:
					print('path not found')
					state = 'shutdown'
				#state = 'shutdown'
				

			elif waypoint == 'resetpoint': #midpoint
				success,path = success,path = find_path(des_pose,des_quat,gripper=False,ignore_collisions=True)
				if success:
					execute_path(path)
					waypoint = grasppoints[obj_id+1]
				else:
					print('path not found')
					state = 'shutdown'
				obj_id += 1
				if obj_id == 3:
					state = 'shutdown'"""





		
