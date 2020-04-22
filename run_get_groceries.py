# state machine goes here
from get_groceries_env import GroceriesEnvironment

env = GroceriesEnvironment()




# 1. best object to pick

obj = env.best_object_to_pick()

if not env.check_for_fallen_objects() == []:
    break
