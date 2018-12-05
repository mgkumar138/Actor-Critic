import gym
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import random
from collections import deque
from matplotlib import pyplot as plt

# https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69

''' Version 0 
- 1 actor & 1 critic -> A2C? 
- policy loss & value loss '''

'''Issues 
- Cartpole: py172, action input to have dimensions (None,2) but got shape (2,1)'''


def preprocess(state):
    process_state = np.mean(state, axis=2).astype(np.uint8)  # compress 3 channels into 1: RGB --> grayscale
    process_state = process_state[::2, ::2]  # downsample pixels by half or crop by tf bounding box
    process_state_size = list(process_state.shape)
    process_state_size.append(1)  # reshape state size into [batch_size=1, state_size] for model
    process_state = np.reshape(process_state, process_state_size)
    return process_state


class A3CAgent:
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess  # ?
        self.lr = 0.00025
        self.eps = 1.0
        self.eps_decay = 0.995
        self.eps_min = 0.01
        self.gamma = 0.99
        self.tau = 0.125  # change from DQN
        self.memory = deque(maxlen=1000000)  # episodic memory to pull from to train model

        self.action_size = self.select_action_size()
        self.state_size = self.select_state_size()

        # ========================================================================= #
        #                               ACTOR MODEL                                 #
        # Objective: state actor model's network parameters.                        #
        # It will change through training.                                          #
        # But critic model is the one to tell us how to change.                     #
        # So chain rule: de/dA = de/DC * dC/dA where e = error                      #
        # ========================================================================= #

        # Get actions based on actor model given state
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        # KEY STEP: obtain de / dC, stored from session: error gradient from critic
        # dc/dA or dA/dC?
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.action_size])

        # [de/dA] neg grad for gradient ascent.
        # tf.grad(A,thetaA, init grad) compute dA/dtheta with initial dA/dC
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad)

        # dA/dthetaA * thetaA
        grads = zip(self.actor_grads, actor_model_weights)

        # compute de/dA & do backprop
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)

        # ============================================================================ #
        #                               CRITIC MODEL                                   #
        # Objective: Calculate output Q given state & action inputs using Critic model #
        # ============================================================================ #

        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        # get grad using model output V(s) & action taken by actor dthetaC/dA
        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)

        # initialise for later gradient calculations
        self.sess.run(tf.global_variables_initializer())

    def select_state_size(self):
        if self.env.observation_space.shape == ():
            state_size = self.env.observation_space.n  # discrete state size
        elif len(self.env.observation_space.shape) == 1:
            state_size = self.env.observation_space.shape[0]  # convert box vector to 1 unit state space
        else:
            process_state = preprocess(self.env.reset())
            state_size = process_state.shape
        return state_size

    def select_action_size(self):
        if self.env.action_space.shape == ():
            action_size = self.env.action_space.n  # discrete state size
        elif len(self.env.action_space.shape) == 1:
            action_size = self.env.action_space.shape[0]  # convert box vector to 1 unit state space
        else:
            print('Error in action size')
        return action_size

    # define dC/dA -> how the change in actor model parameters changes critic valuation
    # update critic model such that y_pred >> y_true hill climbing

    # =========================== #
    # Create models & memory bank #
    # =========================== #

    def create_actor_model(self):
        # actor model involved in utilising policy. need to update what policy (update weights) to use by critic
        # It takes in env space and spits out Probability of action to take based on policy -> weights of network.
        state_input = Input(shape=(self.state_size,), name='state_input')
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        policy_output = Dense(self.action_size, activation='linear', name='policy_output')(h3)

        model = Model(inputs=state_input, outputs=policy_output)
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return state_input, model
        # return state_input to update actor-critic on what state is being updated now.
        # Model contains both input and output values

    def create_critic_model(self):
        # calculate V(s) given state and action Q(s,a). Middle layer combine state and action
        state_input = Input(shape=(self.state_size,), name='state_input')
        state_h1 = Dense(24, activation='relu')(state_input)  # extra layer for env space, as recommended
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=(self.action_size,), name='action_input')
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])  # combine both nets
        merged_h1 = Dense(24, activation='relu')(merged)

        value_output = Dense(1, activation='linear')(merged_h1)  # Output = 1, Q(s,a)

        model = Model(inputs=[state_input, action_input], outputs=value_output)
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return state_input, action_input, model  # to update actor network

    def remember(self, state, action, reward, new_state, done):  # store past experience as a pre-defined table
        self.memory.append([state, action, reward, new_state, done])

        # ===================================== #
        # Training both models using experience #
        # ===================================== #

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)

        self._train_critic(samples)
        self._train_actor(samples)

    def _train_critic(self, samples):  # compute value loss given current state & action taken by actor
        for sample in samples:
            state, action, reward, new_state, done = sample  # Check what is sample
            if done:
                target = reward
            else:
                # given action from actor, what is the reward according to critic?
                target_actor_action = self.target_actor_model.predict(new_state)  # find action by actor
                target = reward + self.gamma * self.target_critic_model.predict([new_state, target_actor_action])[0][0]

            # given curr state and action, train and change critic network weights
            self.critic_model.fit(x={'state_input': state, 'action_input': action}, y=target, verbose=0)  # train critic based on value loss: (R-V(s))**2

    def _train_actor(self, samples):  # compute policy loss using policy from actor & Q value from critic
        for sample in samples:
            state, action, reward, new_state, done = sample
            # given curr state, predict action using actor model
            predicted_action = self.actor_model.predict(state)

            # obtain de/dC grad from critic
            grads = self.sess.run(self.critic_grads,
                                  feed_dict={self.critic_state_input: state,
                                             self.critic_action_input: predicted_action})[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: state,
                self.actor_critic_grad: grads})  # feed into predefined placeholder

            # self.actor_model.fit(state, grads, verbose=0) # train actor policy loss =(-log(policy)(R-V(s)) using de/dC

        # =============================================== #
        #  Update actor and critic target model weights   #
        # =============================================== #

    def train_target(self):
        self._update_actor_target()
        self._update_critic_target()

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = self.tau*actor_model_weights[i] + (1-self.tau)*actor_target_weights[i]
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = self.tau*critic_model_weights[i] + (1-self.tau)*critic_target_weights[i]
        self.target_critic_model.set_weights(critic_target_weights)

    # ====================#
    # Model Prediction    #
    # =================== #

    def act(self, state):
        self.eps *= self.eps_decay
        self.eps = max(self.eps_min, self.eps)
        if np.random.random() < self.eps:
            return self.env.action_space.sample()
        return self.actor_model.predict(state)[0]  # if more than 1 action, use argmax to select action with most prob

        # ============= #
        #  Save models  #
        # ==============#

    def save_model(self, fn):
        self.actor_model.save(fn)
        self.critic_model.save(fn)


def main():
    sess = tf.Session()
    K.set_session(sess)
    env = gym.make("Pendulum-v0")
    agent = A3CAgent(env, sess)

    episodes = int(input('How many episodes?'))
    save_file = input('Save Model? [y/n]: ')
    save_plot = input('Save Plot? [y/n]: ')
    rend_env = input('Render Environment? [y/n]: ')

    time = 250
    batch_size = 32

    tot_r = []

    for e in range(episodes):

        ri = []
        state = env.reset()
        state = state.reshape(1, state.size)

        for t in range(time):

            if rend_env == 'y':
                env.render()

            action = agent.act(state)
            action = action.reshape(1, action.size)

            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(1, state.size)

            agent.remember(state, action, reward, new_state, done)
            agent.replay(batch_size)
            agent.train_target()

            state = new_state

            ri.append(reward)
            print('Action = {}, Reward @ frame = {}'.format(action, reward))

            if done:

                avg_ri = np.mean(ri)
                tot_r.append(avg_ri)

                print('Episode {} of {},avg_ri{}'.format(e, episodes, avg_ri))
                break

    if rend_env == 'y':
        env.close()

    if save_file == 'y':
        agent.save_model('Pendulum_{}epi.h5'.format(episodes))

    plt.figure()
    plt.plot(list(range(1, episodes + 1)), tot_r)
    #plt.plot(list(range(0, 200)), ri)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('AC Pendulum {} Episodes'.format(episodes))

    if save_plot == 'y':
        plt.savefig('Pendulum_{}eps.png'.format(episodes))

    plt.show()

    return tot_r, episodes


if __name__ == "__main__":
    total_reward, epi = main()


''' convolutional layers

self.policy  = FC, out = Dense(action_size)
self.value = FC, Dense(1)

bellman_steps = 4
clip_grad = 0.1
total_env = 64
processes_count  = mp.cpu_count()
env_per_process = math.ceil(total_env/processes_count)'''
