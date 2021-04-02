from utils.core import *
import numpy as np


def interact(agent, world, recipe):
    """Carries out interaction for this agent taking this action in this world.

    The action that needs to be executed is stored in `agent.action`.
    """

    reward = -0.02 * (world.width * world.height * 0.1)

    recipe_name = [x.name for x in recipe][0]
    plate_name = [x.full_plate_name for x in recipe][0]

    # agent does nothing (i.e. no arrow key)
    # if agent.action == (0, 0):
    #     return 0

    action_x, action_y = world.inbounds(tuple(np.asarray(agent.location) + np.asarray(agent.action)))
    gs = world.get_gridsquare_at((action_x, action_y))

    # if floor in front --> move to that square
    if isinstance(gs, Floor): #and gs.holding is None:
        agent.move_to(gs.location)

    # if holding something
    elif agent.holding is not None:
        # if delivery in front --> deliver
        if isinstance(gs, Delivery):
            obj = agent.holding

            if 'Salad' in recipe_name and plate_name == obj.name and obj.is_deliverable():
                gs.acquire(obj)
                agent.release()
                print('\nDelivered {}!'.format(obj.full_name))
                reward = 2 * (world.width * world.height * 0.4)

            elif 'Salad' not in recipe_name and obj.is_deliverable():
                gs.acquire(obj)
                agent.release()
                print('\nDelivered {}!'.format(obj.full_name))
                reward = 2 * (world.width * world.height * 0.4)

        # if occupied gridsquare in front --> try merging
        elif world.is_occupied(gs.location):
            # Get object on gridsquare/counter
            obj = world.get_object_at(gs.location, None, find_held_objects = False)

            if mergeable(agent.holding, obj):
                reward = 2 * (world.width * world.height * 0.4)       
                world.remove(obj)
                o = gs.release() # agent is holding object
                world.remove(agent.holding)
                agent.acquire(obj)
                world.insert(agent.holding)
                # if playable version, merge onto counter first
                if world.arglist.play:
                    gs.acquire(agent.holding)
                    agent.release()


        # if holding something, empty gridsquare in front --> chop or drop
        elif not world.is_occupied(gs.location):
            obj = agent.holding
            if isinstance(gs, Cutboard) and obj.needs_chopped() and not world.arglist.play:
                # normally chop, but if in playable game mode then put down first
                obj.chop()
                reward = 0.5 * (world.width * world.height * 0.1)
                # reward = 5
            elif isinstance(gs, Wall) or isinstance(gs, Cutboard) and not world.arglist.play:
                pass            
            else:
                gs.acquire(obj) # obj is put onto gridsquare
                agent.release()
                #reward = -1 * (world.width * world.height * 0.1)
                assert world.get_object_at(gs.location, obj, find_held_objects =\
                    False).is_held == False, "Verifying put down works"


    # if not holding anything
    elif agent.holding is None:
        # not empty in front --> pick up
        if world.is_occupied(gs.location) and not isinstance(gs, Delivery):
            obj = world.get_object_at(gs.location, None, find_held_objects = False)
            
            # if in playable game mode, then chop raw items on cutting board
            if isinstance(gs, Cutboard) and obj.needs_chopped() and world.arglist.play:
                obj.chop()
                reward = 2 * (world.width * world.height * 0.4)
            else:
                gs.release()
                agent.acquire(obj)
                # ok
                #reward = -1 * (world.width * world.height * 0.1)

        # if empty in front --> interact
        elif not world.is_occupied(gs.location):
            pass

        
    return reward