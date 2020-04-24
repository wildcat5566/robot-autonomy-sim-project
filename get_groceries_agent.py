import numpy as np
import scipy as sp

# # steps to run this in Jupyterlab
# action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION) # See rlbench/action_modes.py for other action modes
# env         = Environment(action_mode, '', ObservationConfig(), False)
# task        = env.get_task(PutGroceriesInCupboard) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
# pose_sensor = NoisyObjectPoseSensor(env)
# agent = StandardAgent(env, task, pose_sensor)

class StandardAgent():

    def __init__(self,env,task,pose_sensor):
        self.env  = env
        self.task = task
        self.pose_sensor = pose_sensor
    
    # AGENT METHODS
    def move_to_safe_spot(self, obj):
        # obj: string 'soup'
        # identify safe spot for the object
        # grasp object
        # move the object
        # release the object
        pass
    
    def move_to_cupboard(self, obj):
        # grasp object
        # identify cupboard position
        # move to cupboard
        # release object
        pass
    
    def push_obj_to_ledge(obj):
        pass

    def grasp(self, obj):
        grasp_pose     = pose_sensor(obj)
        pre_grasp_pose = get_new_offset_EE_position(position, quat, 0.05)
        gripper_close_pose = grasp_pose

        move_to_pose(pre_grasp_pose)
        move_to_pose(grasp_pose)
        move_to_pose(gripper_close_pose)
        
        pass

    def move_to_pose(self,pos,quat,gripper):
        # Given a position and orientation of the gripper, the agent moves the gripper to that pose
        # Inputs:
            # pos: 3x1 vector of x,y,z location in world frame
            # quat : 4x1 vector of quaternion in world frame
            # gripper : boolean (T = CLOSED, F = OPEN)
        # Outputs: 
            # success: a boolean to represent path success
        
        path = self.get_valid_path(pos,quat,gripper)
        success = self.execute_path(path)
        
        return success

    # HELPER METHODS ONLY USED BY THE AGENT------------------------------
    
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
        
        test_quats = self.get_possible_rotations(quat,offset=0)
        i=1

        for q in test_quats:
            success, path = self.find_path(pos,quat,q,gripper,ignore_collisions)
            if success:
                self.execute_path(path)
                print('path found to quat', i)
                break
            else:
                print('path not found to quat', i)
            i+=1

        return success

    def find_path(self,pos,quat,gripper,ignore_collisions=False):
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

	def get_possible_rotations(quat,offset=0):
		# Get rotations about z in 90 degree increments
        # INPUT
            # quaternion : waypoint quaternion
            # offset : how much off of 0 do we want to start doing the 90 degree transformations (for use if first try of this fxn returns no good paths)
        # OUTPUT
            # quats : 4 quaternions representing 0, 90, 180, 270 degree offset in Z from input quaternion
        quats = np.zeros((4,4))
		quats[0] = quat

		matrix = R.from_quat(quat).as_matrix()
		ninety = R.from_rotvec(np.pi/2 + offset * np.array([0, 0, 1])).as_matrix()

		for i in range(1,4):
			matrix = np.dot(ninety,matrix)
			quats[i] = R.from_matrix(matrix).as_quat()

		return quats

	def quat_to_homo_trans_mat(quat_world):
	    A = np.eye(4)
	    A[:3,:3] = R.from_quat(quat_world).as_matrix()
	    return A

	def get_new_offset_EE_position(position, quat, offset_amount):
	    # Move the EE along the z-xis by the specified offset_amount
        # Inputs:
            # pos  quat,: x,y,z of waypoint in world frame
            # quat : quat of waypoint in world frame
            # offset_amount : offset_amount in z
        # Output:
            # new_position : x,y,z of waypoint in world frame
        	    
        # create rotation matrix between world frame and can frame
	    world2gripper        = quat_to_homo_trans_mat(quat)
	    world2gripper[0:3,3] = position
	    
        # create translation vector in gripper frame
	    translation_vec        = np.array([0, 0, offset_amount, 0]).reshape(-1, 1)
	    translation_in_global  = world2gripper @ translation_vec
	    new_position           = translation_in_global.flatten()[:3] + position

	    return new_position

