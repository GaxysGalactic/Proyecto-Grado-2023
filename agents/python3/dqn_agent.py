import gym
import numpy as np
import tensorflow as tf

##################################
#  _______   ______     .__   __. 
# |       \ /  __  \    |  \ |  | 
# |  .--.  |  |  |  |   |   \|  | 
# |  |  |  |  |  |  |   |  . `  | 
# |  '--'  |  `--'  '--.|  |\   | 
# |_______/ \_____\_____\__| \__| 
##################################

# Define the Deep Q Network (DQN) model
class DQNNetwork(tf.keras.Sequential):
    def __init__(self, num_actions):
        super(DQNNetwork, self).__init__([
            tf.keras.layers.Dense(512, activation='relu', name='hidden'),
            tf.keras.layers.Dense(num_actions, activation=None, name='output')
        ])

# Define the DQN Agent class
class MultiUnitDQNAgent:

    actions = ["up", "down", "left", "right", "bomb", "detonate", "nothing"]
    agent_id = "a"

    # Sets the id of the agent
    def set_agent_id(self, new_id: str):
        self.agent_id = new_id

    # INITIALIZER
    def __init__(self, num_units, num_actions_per_unit, replay_memory_size=450, load_model_path=None):
        self.num_units = num_units
        self.num_actions_per_unit = num_actions_per_unit
        self.total_actions = num_actions_per_unit ** num_units
        self.model = DQNNetwork(self.total_actions)
        self.target_model = DQNNetwork(self.total_actions)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = 0.95
        self.batch_size = 32
        self.replay_memory_size = replay_memory_size
        self.replay_memory = []
        self.psummary = False

        # Attempt to load model from file
        if load_model_path:
            try:
                self.model = tf.keras.models.load_model(load_model_path, custom_objects={'DQNNetwork': DQNNetwork})
                self.target_model.set_weights(self.model.get_weights())
                print("Loaded entire model from file!")
            except (OSError, ValueError):
                try:
                    self.model.load_weights(load_model_path)
                    print("Loaded model weights!")
                except (OSError, ValueError):
                    print("Failed to load the model or weights. Continuing with a new model...")

        self.model.compile(optimizer=self.optimizer, loss="mse")

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

    # Select an action based on the current epsilon random chance
    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.total_actions)
        q_values = self.model.predict(state)
        return np.argmax(q_values)

    def get_actions(self, r_state, state, epsilon):
        idx = self.select_action(state, epsilon)
        act_str = self.action_matrix[idx]
        act_li = act_str.split(',')

        my_units = r_state.get("agents").get(self.agent_id).get("unit_ids")
        res = {}
        counter = 0
        for unit_id in my_units:
            res[unit_id] = act_li[counter]
            counter += 1

        return res


    def replay(self):
        if len(self.replay_memory) < self.batch_size:
            return
        
        # Keep the replay memory size capped
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.pop(0)  # Remove the oldest experience

        samples = np.random.choice(len(self.replay_memory), self.batch_size, replace=False)
        for sample in samples:
            state, action, reward, next_state, done = self.replay_memory[sample]

            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(next_state)[0])
                target[0][action] = reward + Q_future * self.gamma

            self.model.fit(state, target, epochs=1, verbose=0)
        
        if not self.psummary:
            self.psummary = True
            print(self.model.input_shape)
            print(self.model.summary())

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
