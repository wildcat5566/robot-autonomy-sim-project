import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
from get_groceries_agent import StandardAgent

# some constants
DEFAULT_ACTION_MODE = ArmActionMode.ABS_EE_POSE_PLAN
DEFAULT_TASK        = PutGroceriesInCupboard
# DEFAULT_POSE_SENSOR = Noisy
DEFAULT_AGENT       = StandardAgent

class NoisyObjectPoseSensor:
    def __init__(self, env):
        self._env = env
        self._pos_scale = [0.005] * 3
        self._rot_scale = [0.01] * 3

    def get_poses(self, noisy = True):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)

        obj_poses = {}
        for obj in objs:
            # get ground truth object poses
            name = obj.get_name()
            pose = obj.get_pose()
            
            # sample normal noise for the poses
            pos, quat_wxyz = sample_normal_pose(self._pos_scale, self._rot_scale)
            gt_quat_wxyz = quaternion(pose[6], pose[3], pose[4], pose[5])
            perturbed_quat_wxyz = quat_wxyz * gt_quat_wxyz
            
            # add noise to the ground truth poses if noisy flag set
            if noisy:
                pose[:3] += pos
                pose[3:] = [perturbed_quat_wxyz.x, perturbed_quat_wxyz.y, perturbed_quat_wxyz.z, perturbed_quat_wxyz.w]
            obj_poses[name] = pose
        
        return obj_poses

class StepFailedError(Exception):
    pass

class GroceriesEnvironment():
    '''
    Sets up the abstracted environment for the task Put Groceries in Cupboard
    '''
    def __init__(self,\
            action_mode=DEFAULT_ACTION_MODE,\
            task=DEFAULT_TASK, \
            agent=DEFAULT_AGENT):
        
        # environment setup
        self.action_mode = action_mode
        self.environment = Environment(action_mode)
        self.agent       = StandardAgent(self.environment, self.task, self.pose_sensor)

        # task setup
        self.task = self.environment.get_task(task)
        self.obs  = None
        self.objects = None

        # pose sensor setup
        self.pose_sensor = NoisyObjectPoseSensor(self.environment)
        self.noisy = False

        # environment parameters
        self.obj_poses = self.get_poses(self.noisy)

        # feature settings
        self.visualize = False
        self.print_debug = False

        # task information
        self.state       = 'initialize'
        self.objects_left_to_pick = self.objects
        self.objects_picked_

    def initialize(self):
        '''Initialize the environment'''
        self.state= 'initialize'
        self.obs  = self.task.reset()[1]
        self.obj_poses = self.get_poses()
    
    def get_poses(self, noisy=self.noisy):
        ''' Returns the object poses dict for all objects '''
        return self.pose_sensor.get_poses(noisy)

    def move(self, pose):
        '''Moves the End Effector to the desired position with the agent'''

        position, quaternion = pose
        pass

    def grasp(self, pose):
        '''Moves EE to the grasp position, grasps. Returns grasp success.'''
        pass
    
    def check_for_fallen_objects(self):
        '''Checks for fallen objects, if found, returns names. If not, returns False.'''
        pass

    def best_object_to_pick(self):
        '''
        Decide on the best object to pick next based on arrangement, accessibility and
        previously picked objects. 
        '''
        pass
    
    