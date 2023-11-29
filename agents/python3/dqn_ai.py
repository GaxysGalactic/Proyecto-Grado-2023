from typing import Dict
from gym import Gym
import os
import random
from random_agent import RandomAgent
from dodger_agent import DodgerAgent
from dqn_agent import MultiUnitDQNAgent
from initial_states import initial_states_li
import utilities
import numpy as np
import logging
import pickle

##################################
#  _______   ______     .__   __. 
# |       \ /  __  \    |  \ |  | 
# |  .--.  |  |  |  |   |   \|  | 
# |  |  |  |  |  |  |   |  . `  | 
# |  '--'  |  `--'  '--.|  |\   | 
# |_______/ \_____\_____\__| \__| 
##################################

async def run_DQN(fwd_model_uri: str):
    # Logging
    logExists = False
    if os.path.isfile("TrainingDQN.log"):
        logExists = True
    logging.basicConfig(filename='TrainingDQN.log', level=logging.INFO, format='%(message)s')
    if not logExists:
        logging.info("Episode, Tick, Reward, Total Reward")

    # Gym Creation
    gym = Gym(fwd_model_uri)
    await gym.connect()
    env = gym.make("bomberland-open-ai-gym", random.choice(initial_states_li))

    # Opponents
    randall = RandomAgent()
    dodgy = DodgerAgent()

    # DQN Bot
    if os.path.isfile("dqn_model.keras"):
        qbot = MultiUnitDQNAgent(3, 7, load_model_path="dqn_model.keras")
    elif os.path.isfile("dqn_weights.keras"):
        qbot = MultiUnitDQNAgent(3, 7, load_model_path="dqn_weights.keras")
    else:
        qbot = MultiUnitDQNAgent(3, 7)

    # Resume epsilon progress if needed / wanted
    if os.path.isfile("epsilon.pickle"):
        with open("epsilon.pickle", "rb") as file:
            epsilon = pickle.load(file)
    else:
        epsilon = 1.0
    epsilon_decay = 0.98 # Fairly aggressive - do note
    min_epsilon = 0.01

    # Main Training Loop - Episodes (Matches)
    for episode in range(10):
        setup = utilities.setup_game()

        # Get important ids
        training_id = setup["Training_id"]
        opponent_id = setup["Opponent_id"]
        opponent = setup["Opponent"]

        # Pass them to our possible opponents
        if opponent == "Random":
            randall.set_agent_id(opponent_id)
        elif opponent == "Dodger":
            dodgy.set_agent_id(opponent_id)
        
        # Setup the AI agent episode
        qbot.set_agent_id(training_id)

        # Get State and parse it for our AI agents
        state = env._initial_state
        c_state = np.asarray(utilities.parse_state(state, training_id))
        c_state = np.reshape(c_state, [1, len(c_state)])

        total_reward = 0
        
        # Main Trianing Loop - Steps (Ticks)
        for time_step in range(450):
            print(time_step)
            actions = []
            
            # DQN Agent
            choice = qbot.select_action(c_state, epsilon)
            q_actions = qbot.get_actions(state, c_state, epsilon)
            for unit in q_actions:
                if q_actions[unit] != "nothing":
                    action = utilities.parse_action(q_actions[unit], unit, training_id, state)
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
                    action = utilities.parse_action(opp_actions[unit], unit, opponent_id, state)
                    if action:
                        actions.append(action)

            # Calculate Next Tick and Reward
            next_state, done, info = await env.step(actions)
            reward = utilities.calculate_reward(state, next_state, training_id, opponent_id, time_step)

            # Speed Bonus for Reward
            if done and time_step < 200 and utilities.team_hp > 0:
                reward += 5

            # Log Rewards
            total_reward += reward
            print("Reward: ")
            print(reward)
            print("Total Reward: ")
            print(total_reward)
            logging.info(str(episode) + "," + str(time_step) + "," + str(reward) + "," + str(total_reward))

            # Parse our new state
            n_state = utilities.parse_state(next_state, training_id)
            n_state = np.reshape(n_state, [1, len(n_state)])

            # Trigger replay and loop states
            qbot.replay_memory.append((c_state, choice, reward, n_state, done))
            qbot.replay()
            state = next_state
            c_state = n_state

            # Update Target Model often
            if time_step % 10 == 0:
                qbot.update_target_model()

            # Make a midway save of the model - partway through a game.
            if time_step % 200 == 0 and time_step > 0:
                qbot.model.save("dqn_model.keras")
                qbot.model.save_weights("dqn_weights.keras")

            if done:
                break
        await env.reset(random.choice(initial_states_li))
        qbot.model.save("dqn_model.keras")
        qbot.model.save_weights("dqn_weights.keras")

        # Decay epsilon and save it
        epsilon *= epsilon_decay
        epsilon = max(min_epsilon, epsilon)
        with open('epsilon.pickle', 'wb') as file: 
            pickle.dump(epsilon, file) 

    await gym.close()