from typing import Dict
from typing import Union

_move_set = set(("up", "down", "left", "right"))

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