import numpy as np
import scipy as sp
from quaternion import from_rotation_matrix, quaternion

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

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
        delta_pos = [(np.random.rand() * 2 - 1) * 0.005, 0, 0] #
        delta_quat = [0, 0, 0, 1]# xyzw
        gripper_pos = [np.random.rand() > 0.5]
        return delta_pos + delta_quat + gripper_pos


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

def go_to_pose(position,quaternion,gripper,ignore_collisions):
    try:
        # ,algorithm=Algos.BiTRRT
        path_class = env._robot.arm.get_path(position=position, quaternion=quaternion,ignore_collisions=ignore_collisions) #pyrep/pyrep/robots/arm.py
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

def check_for_fallen_objects(obj_poses):
    fallen = []
    groceries = ['soup', 'sugar', 'coffee']
    for obj in obj_poses:
        if obj in groceries:
            print(obj, obj_poses[obj])
            qx, qy = obj_poses[obj][3], obj_poses[obj][4]
            if abs(qx) > 0.01 and abs(qy) > 0.01:
                fallen.append(obj)
    return fallen
        
def best_object_to_grasp(obj_poses, gripper_pose, fallen):
    groceries = ['soup', 'sugar', 'coffee']
    if len(fallen) == len(groceries):
        return None 

    valid = [g for g in groceries if g not in fallen]
    print(valid)

    best = None    
    mindist = 100.0
    gripper_xyz = gripper_pose[:3]
    for obj in obj_poses:
        if obj in valid:
            obj_xyz = obj_poses[obj+'_grasp_point'][:3]
            dist = np.sum((np.array(obj_xyz) - np.array(gripper_xyz))**2)
            if mindist > dist:
                mindist = dist
                best = obj
    return best

if __name__ == "__main__":
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION) # See rlbench/action_modes.py for other action modes DELTA_EE_POSE
    env = Environment(action_mode, '', ObservationConfig(), False)
    task = env.get_task(PutGroceriesInCupboard) # available tasks: EmptyContainer, PlayJenga, PutGroceriesInCupboard, SetTheTable
    agent = RandomAgent()
    obj_pose_sensor = NoisyObjectPoseSensor(env)
    descriptions, obs = task.reset()

    obj_poses = obj_pose_sensor.get_poses()
    fallen = check_for_fallen_objects(obj_poses)
    print(fallen)
    #print(descriptions)
    
    gripper_pose = obs.gripper_pose
    target = best_object_to_grasp(obj_poses, gripper_pose, fallen)
    print(target)

    current_joints = obs.joint_positions
    gripper_pose = obs.gripper_pose
    init_gripper_pose = gripper_pose
    wp_q = [0, 1, 0, 0]

    while True:
        # Getting noisy object poses
        obj_poses = obj_pose_sensor.get_poses()
        target = 'soup'
        target_pos = obj_poses[target][:3]

        # Getting various fields from obs
        current_joints = obs.joint_positions
        gripper_pose = obs.gripper_pose


        # i = 0, move to right of object
        print("State: 0, move to pre-position")
        st = False
        while st == False:
            st = go_to_pose([target_pos[0], target_pos[1]-0.15, target_pos[2]], wp_q, 1, ignore_collisions=False)

        # i = 1, start pushing object
        print("State: 1, start pushing object")
        st = False
        while st==False:
            obj_poses = obj_pose_sensor.get_poses()
            target_pos = obj_poses[target][:3]
                
            st = go_to_pose([target_pos[0], 0.6, target_pos[2]], wp_q, 1, ignore_collisions=True)

        # i = 2, reset gripper pose
        st = False
        print("State: 2, reset gripper pose")
        while st == False:
            st = go_to_pose(init_gripper_pose[:3], init_gripper_pose[3:], 1, ignore_collisions=True)

        # if terminate:
        #     break

    env.shutdown()



    """wp_pos = [[gripper_pose[0], gripper_pose[1], target_pos[2]+0.2],
                  [target_pos[0], target_pos[1]-0.15, target_pos[2]+0.2],
                  [target_pos[0], target_pos[1]-0.15, target_pos[2]]]
        wp_q = [0, 1, 0, 0] #gripper_pose[3:] #0100 [-0.7071,0,0,0.7071] [-0.5, 0.5, 0.5, 0.5]

        for i in range(3):
            print("State: {}".format(i))
            st = False
            while st == False:
                st = go_to_pose(wp_pos[i], wp_q, 1, collisions=True)"""


    #xpos, ypos = target_pos[0], target_pos[1]
    """dx, dy = 0., 0.
        while dy + target_pos[1] <= 0.6:
            st = False
            while st==False:
                obj_poses = obj_pose_sensor.get_poses()
                target_pos = obj_poses[target][:3]
                
                st = go_to_pose([target_pos[0], target_pos[1]+dy, target_pos[2]], wp_q, 1, collisions=True)

            #move to next step
            dy += 0.2
            print(target_pos[1]+dy)"""
