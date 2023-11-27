import asyncio
from typing import Dict
from gym import Gym
import os
import random
from random_agent import RandomAgent
from dodger_agent import DodgerAgent
from dqn_agent import DQNNetwork, MultiUnitDQNAgent
from initial_states import initial_states_li
from utilities import parse_action
import numpy as np
import logging

fwd_model_uri = os.environ.get(
    "FWD_MODEL_CONNECTION_STRING") or "ws://127.0.0.1:6969/?role=admin"

mock_state: Dict = {
        "game_id":"dev",
        "agents":{
            "a":{
                "agent_id":"a","unit_ids":["c","e","g"]},
            "b":{"agent_id":"b","unit_ids":["d","f","h"]}},
        "unit_state":{"c":{"coordinates":[5,10],"hp":3,"inventory":{"bombs":3},"blast_diameter":3,"unit_id":"c","agent_id":"a","invulnerability":0},"d":{"coordinates":[9,10],"hp":3,"inventory":{"bombs":3},"blast_diameter":3,"unit_id":"d","agent_id":"b","invulnerability":0},"e":{"coordinates":[4,9],"hp":3,"inventory":{"bombs":3},"blast_diameter":3,"unit_id":"e","agent_id":"a","invulnerability":0},"f":{"coordinates":[10,9],"hp":3,"inventory":{"bombs":3},"blast_diameter":3,"unit_id":"f","agent_id":"b","invulnerability":0},"g":{"coordinates":[3,11],"hp":3,"inventory":{"bombs":3},"blast_diameter":3,"unit_id":"g","agent_id":"a","invulnerability":0},"h":{"coordinates":[11,11],"hp":3,"inventory":{"bombs":3},"blast_diameter":3,"unit_id":"h","agent_id":"b","invulnerability":0}},"entities":[{"created":0,"x":4,"y":5,"type":"m"},{"created":0,"x":10,"y":5,"type":"m"},{"created":0,"x":8,"y":3,"type":"m"},{"created":0,"x":6,"y":3,"type":"m"},{"created":0,"x":13,"y":12,"type":"m"},{"created":0,"x":1,"y":12,"type":"m"},{"created":0,"x":14,"y":2,"type":"m"},{"created":0,"x":0,"y":2,"type":"m"},{"created":0,"x":4,"y":4,"type":"m"},{"created":0,"x":10,"y":4,"type":"m"},{"created":0,"x":9,"y":6,"type":"m"},{"created":0,"x":5,"y":6,"type":"m"},{"created":0,"x":10,"y":0,"type":"m"},{"created":0,"x":4,"y":0,"type":"m"},{"created":0,"x":3,"y":4,"type":"m"},{"created":0,"x":11,"y":4,"type":"m"},{"created":0,"x":1,"y":6,"type":"m"},{"created":0,"x":13,"y":6,"type":"m"},{"created":0,"x":14,"y":4,"type":"m"},{"created":0,"x":0,"y":4,"type":"m"},{"created":0,"x":1,"y":13,"type":"m"},{"created":0,"x":13,"y":13,"type":"m"},{"created":0,"x":8,"y":7,"type":"m"},{"created":0,"x":6,"y":7,"type":"m"},{"created":0,"x":10,"y":13,"type":"m"},{"created":0,"x":4,"y":13,"type":"m"},{"created":0,"x":2,"y":12,"type":"m"},{"created":0,"x":12,"y":12,"type":"m"},{"created":0,"x":9,"y":2,"type":"m"},{"created":0,"x":5,"y":2,"type":"m"},{"created":0,"x":9,"y":13,"type":"m"},{"created":0,"x":5,"y":13,"type":"m"},{"created":0,"x":0,"y":0,"type":"m"},{"created":0,"x":14,"y":0,"type":"m"},{"created":0,"x":10,"y":12,"type":"m"},{"created":0,"x":4,"y":12,"type":"m"},{"created":0,"x":1,"y":10,"type":"m"},{"created":0,"x":13,"y":10,"type":"m"},{"created":0,"x":11,"y":6,"type":"m"},{"created":0,"x":3,"y":6,"type":"m"},{"created":0,"x":4,"y":2,"type":"m"},{"created":0,"x":10,"y":2,"type":"m"},{"created":0,"x":12,"y":4,"type":"m"},{"created":0,"x":2,"y":4,"type":"m"},{"created":0,"x":12,"y":8,"type":"m"},{"created":0,"x":2,"y":8,"type":"m"},{"created":0,"x":11,"y":12,"type":"m"},{"created":0,"x":3,"y":12,"type":"m"},{"created":0,"x":12,"y":9,"type":"m"},{"created":0,"x":2,"y":9,"type":"m"},{"created":0,"x":8,"y":2,"type":"w","hp":1},{"created":0,"x":6,"y":2,"type":"w","hp":1},{"created":0,"x":11,"y":3,"type":"w","hp":1},{"created":0,"x":3,"y":3,"type":"w","hp":1},{"created":0,"x":11,"y":13,"type":"w","hp":1},{"created":0,"x":3,"y":13,"type":"w","hp":1},{"created":0,"x":4,"y":14,"type":"w","hp":1},{"created":0,"x":10,"y":14,"type":"w","hp":1},{"created":0,"x":13,"y":2,"type":"w","hp":1},{"created":0,"x":1,"y":2,"type":"w","hp":1},{"created":0,"x":2,"y":1,"type":"w","hp":1},{"created":0,"x":12,"y":1,"type":"w","hp":1},{"created":0,"x":14,"y":13,"type":"w","hp":1},{"created":0,"x":0,"y":13,"type":"w","hp":1},{"created":0,"x":10,"y":11,"type":"w","hp":1},{"created":0,"x":4,"y":11,"type":"w","hp":1},{"created":0,"x":14,"y":14,"type":"w","hp":1},{"created":0,"x":0,"y":14,"type":"w","hp":1},{"created":0,"x":2,"y":3,"type":"w","hp":1},{"created":0,"x":12,"y":3,"type":"w","hp":1},{"created":0,"x":1,"y":3,"type":"w","hp":1},{"created":0,"x":13,"y":3,"type":"w","hp":1},{"created":0,"x":12,"y":10,"type":"w","hp":1},{"created":0,"x":2,"y":10,"type":"w","hp":1},{"created":0,"x":9,"y":5,"type":"w","hp":1},{"created":0,"x":5,"y":5,"type":"w","hp":1},{"created":0,"x":3,"y":8,"type":"w","hp":1},{"created":0,"x":11,"y":8,"type":"w","hp":1},{"created":0,"x":6,"y":6,"type":"w","hp":1},{"created":0,"x":8,"y":6,"type":"w","hp":1},{"created":0,"x":0,"y":7,"type":"w","hp":1},{"created":0,"x":14,"y":7,"type":"w","hp":1},{"created":0,"x":5,"y":8,"type":"w","hp":1},{"created":0,"x":9,"y":8,"type":"w","hp":1},{"created":0,"x":8,"y":13,"type":"w","hp":1},{"created":0,"x":6,"y":13,"type":"w","hp":1},{"created":0,"x":11,"y":1,"type":"w","hp":1},{"created":0,"x":3,"y":1,"type":"w","hp":1},{"created":0,"x":2,"y":6,"type":"w","hp":1},{"created":0,"x":12,"y":6,"type":"w","hp":1},{"created":0,"x":8,"y":5,"type":"w","hp":1},{"created":0,"x":6,"y":5,"type":"w","hp":1},{"created":0,"x":10,"y":10,"type":"w","hp":1},{"created":0,"x":4,"y":10,"type":"w","hp":1},{"created":0,"x":10,"y":6,"type":"w","hp":1},{"created":0,"x":4,"y":6,"type":"w","hp":1},{"created":0,"x":8,"y":8,"type":"w","hp":1},{"created":0,"x":6,"y":8,"type":"w","hp":1},{"created":0,"x":8,"y":10,"type":"w","hp":1},{"created":0,"x":6,"y":10,"type":"w","hp":1},{"created":0,"x":12,"y":0,"type":"w","hp":1},{"created":0,"x":2,"y":0,"type":"w","hp":1},{"created":0,"x":11,"y":7,"type":"w","hp":1},{"created":0,"x":3,"y":7,"type":"w","hp":1},{"created":0,"x":11,"y":2,"type":"w","hp":1},{"created":0,"x":3,"y":2,"type":"w","hp":1},{"created":0,"x":1,"y":0,"type":"o","hp":3},{"created":0,"x":13,"y":0,"type":"o","hp":3},{"created":0,"x":9,"y":12,"type":"o","hp":3},{"created":0,"x":5,"y":12,"type":"o","hp":3},{"created":0,"x":13,"y":7,"type":"o","hp":3},{"created":0,"x":1,"y":7,"type":"o","hp":3},{"created":0,"x":14,"y":11,"type":"o","hp":3},{"created":0,"x":0,"y":11,"type":"o","hp":3},{"created":0,"x":10,"y":1,"type":"o","hp":3},{"created":0,"x":4,"y":1,"type":"o","hp":3},{"created":0,"x":0,"y":1,"type":"o","hp":3},{"created":0,"x":14,"y":1,"type":"o","hp":3},{"created":0,"x":5,"y":1,"type":"o","hp":3},{"created":0,"x":9,"y":1,"type":"o","hp":3}],"world":{"width":15,"height":15},"tick":0,"config":{"tick_rate_hz":10,"game_duration_ticks":300,"fire_spawn_interval_ticks":2}}

agent_ids = ["a", "b"]
opponent_choices = ["Random", "DoNothing", "Dodger"]

team_hp = 9
ex_grid = np.zeros([15,15])

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
        ex_grid[coords] = 1
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
        if p_state["unit_state"][unit]["hp"] - c_state["unit_state"][unit]["hp"] == 1 and c_state["unit_state"][unit]["hp"] == 0:
            reward += 3

    # Surviving Ring of Fire rewards
    if tick > 200 and tick % 50 == 0:
        reward += 1

    # Exploration reward

    return reward


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

async def main():
    # Logging
    logging.basicConfig(filename='Training.log', level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.info("Episode, Tick, Reward")

    gym = Gym(fwd_model_uri)
    await gym.connect()
    env = gym.make("bomberland-open-ai-gym", random.choice(initial_states_li))

    # Opponents
    randall = RandomAgent()
    dodgy = DodgerAgent()

    # DQN Bot
    qbot = MultiUnitDQNAgent(3, 7)
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01

    for episode in range(10):
        setup = setup_game()
        training_id = setup["Training_id"]
        opponent_id = setup["Opponent_id"]
        opponent = setup["Opponent"]

        if opponent == "Random":
            randall.set_agent_id(opponent_id)
        elif opponent == "Dodger":
            dodgy.set_agent_id(opponent_id)
        
        qbot.set_agent_id(training_id)
        state = env._initial_state
        c_state = parse_state(state, training_id)
        c_state = np.reshape(c_state, [1, len(c_state)])
        
        for time_step in range(450):
            print(time_step)
            actions = []
            
            # DQN Agent
            choice = qbot.select_action(c_state, epsilon)
            q_actions = qbot.get_actions(state, c_state, epsilon)
            for unit in q_actions:
                if q_actions[unit] != "nothing":
                    action = parse_action(q_actions[unit], unit, training_id, state)
                    if action:
                        actions.append(action)

            # Opponent
            if opponent == "Random":
                opp_actions = randall.get_actions(state)
            elif opponent == "Dodger":
                opp_actions = dodgy.get_actions(state)
            else:
                opp_actions = []
            for unit in opp_actions:
                if opp_actions[unit] != "nothing":
                    action = parse_action(opp_actions[unit], unit, opponent_id, state)
                    if action:
                        actions.append(action)

            next_state, done, info = await env.step(actions)
            reward = calculate_reward(state, next_state, training_id, opponent_id, time_step)

            # Speed Bonus
            if done and time_step < 200 and team_hp > 0:
                reward += 5

            print("Reward: ")
            print(reward)

            logging.info(str(episode) + "," + str(time_step) + "," + str(reward))

            n_state = parse_state(next_state, training_id)
            n_state = np.reshape(n_state, [1, len(n_state)])

            qbot.replay_memory.append((c_state, choice, reward, n_state, done))
            qbot.replay()
            state = next_state
            c_state = n_state

            if time_step % 10 == 0:
                qbot.update_target_model()

            if done:
                break
        await env.reset(random.choice(initial_states_li))
        qbot.model.save("dqn_model.h5")
        qbot.model.save_weights("dqn_weights.h5")
        epsilon *= epsilon_decay
        epsilon = max(min_epsilon, epsilon)
    await gym.close()


if __name__ == "__main__":
    asyncio.run(main())
