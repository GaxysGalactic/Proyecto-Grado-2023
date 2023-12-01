from typing import Dict
from gym import Gym
import os
import random
from random_agent import RandomAgent
from dodger_agent import DodgerAgent
from initial_states import initial_states_li
import utilities
import numpy as np
import logging
import neat
from asyncPop import AsyncPop
from ai_flag import neat_load_checkpoint, neat_checkpoint_fp
import pickle

#############################################
# .__   __.  _______     ___   .___________.
# |  \ |  | |   ____|   /   \  |           |
# |   \|  | |  |__     /  ^  \ `---|  |----`
# |  . `  | |   __|   /  /_\  \    |  |     
# |  |\   | |  |____ /  _____  \   |  |     
# |__| \__| |_______/__/     \__\  |__|     
#############################################
class NeatAI:

    actions = ["up", "down", "left", "right", "bomb", "detonate", "nothing"]

    def __init__(self):
        # Logging
        logExists = False
        if os.path.isfile("TrainingNEAT.log"):
            logExists = True
        logging.basicConfig(filename='TrainingNEAT.log', level=logging.INFO, format='%(message)s')
        if not logExists:
            logging.info("Genome_ID, Total Reward")

        # Opponents
        self.randall = RandomAgent()
        self.dodgy = DodgerAgent()

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

    # Get the action list for the units based on the given index
    def get_neat_actions(self, r_state, training_id, idx):
        act_str = self.action_matrix[idx]
        act_li = act_str.split(',')

        my_units = r_state.get("agents").get(training_id).get("unit_ids")
        res = {}
        counter = 0
        for unit_id in my_units:
            res[unit_id] = act_li[counter]
            counter += 1

        return res

    # NEAT-python eval_genomes function
    async def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:

            setup = utilities.setup_game()

            # Get important ids
            training_id = setup["Training_id"]
            opponent_id = setup["Opponent_id"]
            opponent = setup["Opponent"]

            # Pass them to our possible opponents
            if opponent == "Random":
                self.randall.set_agent_id(opponent_id)
            elif opponent == "Dodger":
                self.dodgy.set_agent_id(opponent_id)

            # Create network
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            # Get state and parse it
            await self.env.reset(random.choice(initial_states_li))
            state = self.env._initial_state
            c_state = np.asarray(utilities.parse_state(state, training_id))

            total_reward = 0

            for time_step in range(450):  # Run for a maximum of 450 steps (Max ticks in game)
                actions = []

                # Opponent
                if opponent == "Random":
                    opp_actions = self.randall.get_actions(state)
                elif opponent == "Dodger":
                    opp_actions = self.dodgy.get_actions(state)
                else:
                    opp_actions = []
                for unit in opp_actions:
                    if opp_actions[unit] != "nothing":
                        action = utilities.parse_action(opp_actions[unit], unit, opponent_id, state)
                        if action:
                            actions.append(action)

                # NEAT agent actions
                action_probabilities = net.activate(c_state)
                maximum = max(action_probabilities)
                actionidx = random.choice([i for i in range(len(action_probabilities)) if action_probabilities[i] == maximum])
                n_actions = self.get_neat_actions(state, training_id, actionidx)
                for unit in n_actions:
                    if n_actions[unit] != "nothing":
                        action = utilities.parse_action(n_actions[unit], unit, training_id, state)
                        if action:
                            actions.append(action)

                # Calculate Next Tick and Reward
                next_state, done, info = await self.env.step(actions)
                reward = utilities.calculate_reward(state, next_state, training_id, opponent_id, time_step, done)
                total_reward += reward

                state = next_state
                c_state = np.asarray(utilities.parse_state(state, training_id))

                if done:
                    break
            logging.info(str(genome_id) + "," + str(total_reward))
            genome.fitness = total_reward

    async def run_NEAT(self, fwd_model_uri: str):
        # Gym Creation
        self.gym = Gym(fwd_model_uri)
        await self.gym.connect()
        self.env = self.gym.make("bomberland-open-ai-gym", random.choice(initial_states_li))

        # NEAT Config
        config_path = 'neat-config.txt'
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
        
        # Create a Population and add Reporters
        p = AsyncPop(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(10))

        # Restore from checkpoint if needed
        if neat_load_checkpoint:
            p = neat.Checkpointer.restore_checkpoint(neat_checkpoint_fp)

        # Run NEAT algorithm
        winner = await p.run(self.eval_genomes, 100)

        # Best genome found!
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        with open('best_genome.txt', 'w') as f:
            f.write(str(winner_net))
        pickle.dump(winner, open('winner.pickle', 'wb'))

        await self.gym.close()