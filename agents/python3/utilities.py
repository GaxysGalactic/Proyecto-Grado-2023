from typing import Dict
from typing import Union
import numpy as np
import random

_move_set = set(("up", "down", "left", "right"))

team_hp = 9
ex_grid = np.zeros([15,15])

agent_ids = ["a", "b"]
opponent_choices = ["Random", "DoNothing", "Dodger"]

# Setup for a game
def setup_game():
    random.shuffle(agent_ids)
    setup = {
        "Training_id": agent_ids[0],
        "Opponent_id": agent_ids[1],
        "Opponent": random.choice(opponent_choices)
    }
    global team_hp
    team_hp = 9
    global ex_grid
    ex_grid = np.zeros([15, 15])
    return setup

###############################################################################
#      ___       ______ .___________. __    ______   .__   __.      _______.
#     /   \     /      ||           ||  |  /  __  \  |  \ |  |     /       |
#    /  ^  \   |  ,----'`---|  |----`|  | |  |  |  | |   \|  |    |   (----`
#   /  /_\  \  |  |         |  |     |  | |  |  |  | |  . `  |     \   \    
#  /  _____  \ |  `----.    |  |     |  | |  `--'  | |  |\   | .----)   |   
# /__/     \__\ \______|    |__|     |__|  \______/  |__| \__| |_______/    
###############################################################################

def parse_move(move: str, unit_id: str):
    if move in _move_set:
        packet = {"type": "move", "move": move, "unit_id": unit_id}
        return packet

def parse_bomb(unit_id: str):
    packet = {"type": "bomb", "unit_id": unit_id}
    return packet

def parse_detonate(x, y, unit_id: str):
    packet = {"type": "detonate", "coordinates": [
        x, y], "unit_id": unit_id}
    return packet

# returns coordinates of the first bomb placed by a unit
def _get_bomb_to_detonate(unit, raw_state: Dict) -> Union[int, int] or None:
    entities = raw_state.get("entities")
    bombs = list(filter(lambda entity: entity.get(
        "unit_id") == unit and entity.get("type") == "b", entities))
    bomb = next(iter(bombs or []), None)
    if bomb != None:
        return [bomb.get("x"), bomb.get("y")]
    else:
        return None

def parse_action(action: str, unit_id: str, agent_id:str, raw_state: Dict):
    if action in ["up", "left", "right", "down"]:
        return {
            "agent_id": agent_id,
            "action": parse_move(action, unit_id)
        }
    elif action == "bomb":
        return {
            "agent_id": agent_id,
            "action": parse_bomb(unit_id)
        }
    elif action == "detonate":
        bomb_coordinates = _get_bomb_to_detonate(unit_id, raw_state)
        if bomb_coordinates != None:
            x, y = bomb_coordinates
            return {
                "agent_id": agent_id,
                "action": parse_detonate(x, y, unit_id)
            }
    elif action == "nothing":
        return {}
    else:
        print(f"Unhandled action: {action} for unit {unit_id}")

###############################################################################
# .______       ___________    __    ____  ___      .______       _______  
# |   _  \     |   ____\   \  /  \  /   / /   \     |   _  \     |       \ 
# |  |_)  |    |  |__   \   \/    \/   / /  ^  \    |  |_)  |    |  .--.  |
# |      /     |   __|   \            / /  /_\  \   |      /     |  |  |  |
# |  |\  \----.|  |____   \    /\    / /  _____  \  |  |\  \----.|  '--'  |
# | _| `._____||_______|   \__/  \__/ /__/     \__\ | _| `._____||_______/                                                                     
###############################################################################


def calculate_reward(p_state: Dict, c_state: Dict, training_id, opponent_id, tick):
    friendlies = p_state.get("agents").get(training_id).get("unit_ids")
    enemies = p_state.get("agents").get(opponent_id).get("unit_ids")
    entities = p_state.get("entities")
    global team_hp
    global ex_grid
    reward = 0

    # Friendlies
    for unit in friendlies:
        # Exploration Rewards
        coords = c_state["unit_state"][unit]["coordinates"]
        if ex_grid[coords[0], coords[1]] == 0.0 and tick < 200:
            ex_grid[coords[0], coords[1]] = 1
            if np.sum(ex_grid) % 20 == 0:
                reward += 1

        # HP Penalties
        reward += c_state["unit_state"][unit]["hp"] - p_state["unit_state"][unit]["hp"]
        if c_state["unit_state"][unit]["hp"] - p_state["unit_state"][unit]["hp"] == -1:
            team_hp -= 1
            # Was this damage self inflicted?
            sf_fire = list(filter(lambda entity: entity.get(
                "unit_id") in friendlies and entity.get("type") == "x" and entity.get("x") == coords[0] and entity.get("y") == coords[1], entities))
            if sf_fire:
                reward -= 1
            # Death Punishment
            if c_state["unit_state"][unit]["hp"] == 0:
                reward -= 2

    # HP Rewards
    for unit in enemies:
        reward += p_state["unit_state"][unit]["hp"] - c_state["unit_state"][unit]["hp"]
        # Kill reward
        if p_state["unit_state"][unit]["hp"] - c_state["unit_state"][unit]["hp"] == 1:
            # Was this damage caused by friendlies?
            sf_fire = list(filter(lambda entity: entity.get(
                "unit_id") in friendlies and entity.get("type") == "x" and entity.get("x") == coords[0] and entity.get("y") == coords[1], entities))
            if sf_fire:
                reward += 1
            if c_state["unit_state"][unit]["hp"] == 0:
                reward += 3

    # Surviving Ring of Fire rewards
    if tick > 200 and tick % 50 == 0:
        reward += 1

    # Exploration reward

    return reward

###############################################################################
#      _______.___________.    ___   .___________. _______ 
#     /       |           |   /   \  |           ||   ____|
#    |   (----`---|  |----`  /  ^  \ `---|  |----`|  |__   
#     \   \       |  |      /  /_\  \    |  |     |   __|  
# .----)   |      |  |     /  _____  \   |  |     |  |____ 
# |_______/       |__|    /__/     \__\  |__|     |_______|
###############################################################################


def parse_unit_id(uid: str):
    if uid == "c":
        return 0
    elif uid == "e":
        return 1
    elif uid == "g":
        return 2
    elif uid == "d":
        return 3
    elif uid == "f":
        return 4
    elif uid == "h":
        return 5

def parse_units(li: list, unit_state: Dict, friendly_id: str):
    for unit in unit_state:
        stats: Dict = unit_state[unit]
        coordinates = stats["coordinates"]
        coord: float = (coordinates[1]*15 + coordinates[0]) / 225
        hp: float = stats["hp"] / 3
        b_diameter: float = stats["blast_diameter"] / 15
        unit_id: float = parse_unit_id(stats["unit_id"]) / 5
        if stats["agent_id"] == friendly_id:
            agent_id: float = 1
        else:
            agent_id: float = 0
        invuln: float = stats["invulnerable"] / 428
        stun: float = stats["stunned"] / 428
        li.extend([coord, hp, b_diameter, unit_id, agent_id, invuln, stun])
    return li

def parse_entity_type(tp: str):
    if tp == "a":
        return 0
    elif tp == "b":
        return 1
    elif tp == "x":
        return 2
    elif tp == "bp":
        return 3
    elif tp == "fp":
        return 4
    elif tp == "m":
        return 5
    elif tp == "o":
        return 6
    elif tp == "w":
        return 7

def parse_entities(li: list, entities: Dict):
    counter = 0
    for stats in entities:
        counter += 1
        created: float = stats["created"] / 428
        coord: float = (stats["y"]*15 + stats["x"]) / 225
        en_type: float = parse_entity_type(stats["type"]) / 7

        if "hp" in stats:
            hp: float = stats["hp"] / 3
        else:
            hp: float = 0

        if "unit_id" in stats:
            unit_id: float = parse_unit_id(stats["unit_id"]) / 5
        else:
            unit_id: float = 0

        if "expires" in stats:
            expires: float = stats["expires"] / 428
        else:
            expires: float = 0

        if "b_diameter" in stats:
            b_diameter: float = stats["blast_diameter"] / 15
        else:
            b_diameter: float = 0
        
        li.extend([created, coord, en_type, unit_id, expires, hp, b_diameter])

    for i in range(counter, 225):
        li.extend([0, 0, 0, 0, 0, 0, 0])
    
    return li
        

def parse_state(state: Dict, friendly_id):
    unit_state: Dict = state["unit_state"]
    entities: Dict = state["entities"]

    res = []
    res = parse_units(res, unit_state, friendly_id)
    res = parse_entities(res, entities)
    return res