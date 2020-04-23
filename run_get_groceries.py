'''
    This file contains the state machine for the 16-662 project. 
    Contributors: Rohan, Kevin, Angela

    State Machine Sequence:
    0. Initialize
    1. Initial Task Planning:
        - Which object to pickup
    2. For each object
        - Move to object
        - Grasp Object
        - Move to cupboard
        - Release
    3. Reset Environment
        - Move upright objects to safe location
        - For each fallen object:
            - Sweep fallen object to ledge
            - Grasp fallen object and put in safe space
        - Move objects from cupboard to safe location
        - Move all objects from safe space to random spots
    4. Restart from 0.

Handling failure cases with custom exceptions:
1. 
'''
from get_groceries_env import GroceriesEnvironment

env = GroceriesEnvironment()
env.objects = ['soup', 'sugar', 'coffee']

while True: # keep running the state machine loop in perpetuity
    
    #---------------0. Initialize-------------
    env.initialize()
    
    #---------loop over all objects to pick-------
    while len(env.objects_left_to_pick) > 0:
        
        try:
            #---------1. Initial Task Planning--------
            obj = env.best_object_to_pick()

            #---2. Execute plan for each object-------
            env.move_to_object(obj)
            env.grasp_object(obj)
            env.move_to_cupboard(obj)
            env.release_object(obj)
        
        except StepFailedError:
            break
    
    try:
        env.move_upright_objects_to_safe_location()
        env.move_cupboard_objects_to_safe_location()
        env.move_fallen_objects_to_safe_location()
        env.move_safe_objects_to_random_locations()
    except:
        break

    pass
