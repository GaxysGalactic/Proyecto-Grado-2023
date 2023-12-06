from typing import Union
from game_state import GameState
import asyncio
import random
import os
import numpy as np
import utilities
from dqn_agent import MultiUnitDQNAgent
import tensorflow as tf
from ai_flag import ai_flag
import pickle
import neat
from typing import Dict

uri = os.environ.get(
    'GAME_CONNECTION_STRING') or "ws://127.0.0.1:3000/?role=agent&agentId=agentId&name=defaultName"

class Agent():

    actions = ["up", "down", "left", "right", "bomb", "detonate", "nothing"]

    def __init__(self):
        self._client = GameState(uri)

        # any initialization code can go here

        # Create the action matrix needed to convert index into set of actions
        self.action_matrix = []
        counter = 0
        for action in self.actions:
            for action2 in self.actions:
                for action3 in self.actions:
                    ac = ""
                    ac += action + ","
                    ac += action2 + ","
                    ac += action3
                    self.action_matrix.append(ac)
                    counter += 1

        if ai_flag == "NEAT":
            # NEAT Config
            config_path = 'neat-config.txt'
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
            # Get Winner Net
            with open(r"winner.pickle", "rb") as input_file: 
                winner = pickle.load(input_file)
            winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
            self.model = winner_net
        else:
            self.model = self.get_dqn_model(343)
            self.model.load_weights("dqn_weights.h5")

        self._client.set_game_tick_callback(self._on_game_tick)

        loop = asyncio.get_event_loop()
        connection = loop.run_until_complete(self._client.connect())
        tasks = [
            asyncio.ensure_future(self._client._handle_messages(connection)),
        ]
        loop.run_until_complete(asyncio.wait(tasks))

    def get_action_idx(self, p_state):
        if ai_flag == "NEAT":
            outputs = self.model.activate(p_state)
        else: 
            outputs = self.model.predict(p_state)

        maximum = max(outputs)
        actionidx = random.choice([i for i in range(len(outputs)) if outputs[i] == maximum])
        return actionidx

    def get_actions(self, state, training_id, idx):
        act_str = self.action_matrix[idx]
        act_li = act_str.split(',')

        my_units = state.get("agents").get(training_id).get("unit_ids")
        res = {}
        counter = 0
        for unit_id in my_units:
            res[unit_id] = act_li[counter]
            counter += 1

        return res
    
    def get_dqn_model(self, num_actions):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(512, activation='relu', name='hidden'))
        model.add(tf.keras.layers.Dense(num_actions, activation=None, name='output'))
        return model
    
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

    ########################    
    ###       TICK       ###
    ########################

    async def _on_game_tick(self, tick_number, game_state):

        # get my units
        my_agent_id = game_state.get("connection").get("agent_id")
        my_units = game_state.get("agents").get(my_agent_id).get("unit_ids")
        
        parsed_state = np.asarray(utilities.parse_state(game_state, my_agent_id))
        if ai_flag == "DQN":       
            parsed_state = np.reshape(parsed_state, [1, len(parsed_state)])

        actionidx = self.get_action_idx(parsed_state)
        actionlist = self.get_actions(game_state, my_agent_id, actionidx)

        # send each unit an action
        for unit_id in my_units:

            action = actionlist[unit_id]

            # Send Action Logic
            if action in ["up", "left", "right", "down"]:
                await self._client.send_move(action, unit_id)
            elif action == "bomb":
                await self._client.send_bomb(unit_id)
            elif action == "detonate":
                bomb_coordinates = self._get_bomb_to_detonate(unit_id)
                if bomb_coordinates != None:
                    x, y = bomb_coordinates
                    await self._client.send_detonate(x, y, unit_id)
            else:
                print(f"Unhandled action: {action} for unit {unit_id}")


def main():
    Agent()


if __name__ == "__main__":
    main()
