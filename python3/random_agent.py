from typing import Dict
import random

class RandomAgent():

    actions = ["up", "down", "left", "right", "bomb", "detonate"]
    
    agent_id = "a"

    def set_agent_id(self, new_id: str):
        self.agent_id = new_id


    def get_actions(self, raw_state: Dict):
        my_units = raw_state.get("agents").get(self.agent_id).get("unit_ids")

        res = {}

        for unit_id in my_units:
            action = random.choice(self.actions)
            res[unit_id] = action

        return res