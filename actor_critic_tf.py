import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import time as dt
import os


class ActorCriticModel(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCriticModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.logits = tf.keras.layers.Dense(action_size)
        self.values = tf.keras.layers.Dense(1)

    def call(self, inputs):
        A1 = self.dense1(inputs)
        A2 = self.dense2(A1)
        logits = self.logits(A2)

        C1 = self.dense1(inputs)
        C2 = self.dense2(C1)
        values = self.values(C2)
        return logits, values


class ACagent:

    def __init__(self, env):
        self.gamma = 0.95
        self.lr = 0.001
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        self.model = ActorCriticModel(self.state_size, self.action_size)
        self.opt = tf.optimizers.Adam(self.lr)

    def replay(self, memory):
        discount_reward = self.discount_normalise_rewards(memory.rewards)

        with tf.GradientTape() as tape:
            policy_loss, value_loss, total_loss = self.compute_loss(memory, discount_reward)

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.model.trainable_weights)) # train parameters based on gradient

        return policy_loss, value_loss, total_loss

    def act(self, state):
        action_dist, _ = self.model(tf.cast(state[None, :], dtype=tf.float32))
        action_prob_dist = tf.nn.softmax(action_dist)
        return action_prob_dist.numpy(), np.random.choice(range(self.action_size), p=action_prob_dist.numpy()[0])

    def discount_normalise_rewards(self, rewards):
        # determine discount rewards only when done. if not done. cumulative = critic_model.predict(new_state)
        discounted_rewards = []
        cumulative = 0
        for reward in rewards[::-1]:
            cumulative = reward + self.gamma * cumulative
            discounted_rewards.append(cumulative)
        discounted_rewards.reverse()  # need to normalise?

        return discounted_rewards

    def compute_loss(self, memory, discounted_rewards):
        logit, values = self.model(tf.convert_to_tensor(np.vstack(memory.states), dtype=tf.float32))

        # distinguish AC to A2C. AC -> use only values/discounted rewards. Advantage = Q(s,a) - V(s)
        advantage = tf.cast(np.array(discounted_rewards), dtype=tf.float32) - values[:,0]

        value_loss = advantage**2

        # compute actor policy loss with critic input as advantage
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=np.array(memory.actions))

        policy_loss = neg_log_prob * tf.stop_gradient(advantage) # becomes NxN tensor

        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=tf.nn.softmax(logit))

        policy_loss_minent = policy_loss - 0.01 * entropy

        # merge both losses to train network tgt
        comb_loss = tf.reduce_mean((0.5 * value_loss + policy_loss_minent))

        return policy_loss, value_loss, comb_loss


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

def main(episodes):

    env = gym.make("CartPole-v0")
    agent = ACagent(env)
    mem = Memory()

    print('++ Training started on {} at {} ++'.format(dt.strftime('%d%m%Y'), dt.strftime('%H.%M.%S')))
    start_time = dt.time()

    time = 200
    tot_r = []
    tot_loss = []

    for e in range(episodes):
        mem.clear() # clear memory for each episode

        state = env.reset()

        for t in range(time):

            act_prob_dist, action = agent.act(state)

            new_state, reward, done, _ = env.step(action)

            mem.store(state, action, reward)

            state = new_state

            if done:
                policy_loss, value_loss, total_loss = agent.replay(memory=mem)
                tot_loss.append(total_loss.numpy())
                tot_r.append(t)
                mem.clear()
                break

        meanR = np.mean(tot_r)
        maxR = np.amax(tot_r)
        print('E {}/{}, last for {}s, Mean R {:.4f}, Max R {:.4f}, Tr Loss {:.4f}'.format(
            e, episodes, tot_r[-1], meanR,maxR, tot_loss[-1]))

    date = dt.strftime('%d%m%Y')
    clock = dt.strftime('%H.%M.%S')
    print('Training ended on {} at {}'.format(date, clock))
    run_time = dt.time() - start_time
    print('Total Training time: %d Hrs %d Mins $d s' % (run_time // 3600, (run_time % 3600) // 60),
          (run_time % 3600) % 60 // 1)

    '''
    plt.figure()
    plt.plot(list(range(1, episodes + 1)), tot_r)
    plt.plot(list(range(1, episodes + 1)), np.convolve(tot_r, np.ones(100) / 100, mode='same'))
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('A2C CartPole {}eps {} {}'.format(episodes, date, clock))
    plt.show()
    '''
    return tot_r, tot_loss, run_time


if __name__ == "__main__":
    strap_reward, strap_loss, strap_run = [],[],[]

    episodes = 2000
    bootstrap = 1
    for i in range(bootstrap):
        print('Bootstrap {} of {}'.format(i+1, bootstrap))
        total_reward, total_loss, runtime = main(episodes=episodes)
        strap_reward.append(total_reward)
        strap_loss.append(total_loss)
        strap_run.append(runtime)
        #plt.plot(list(range(1, episodes + 1)), strap_reward[i])
    plt.figure()
    plt.subplot(211)
    plt.plot(list(range(1, episodes + 1)), np.mean(np.array(strap_reward),0))
    plt.plot(list(range(1, episodes + 1)),
             np.convolve(np.mean(np.array(strap_reward),0), np.ones(100) / 100, mode='same'))
    plt.title('A2C CartPole reward, Total Runtime {}s'.format(np.round(np.sum(strap_run),3)))

    plt.subplot(212)
    plt.plot(list(range(1, episodes + 1)), np.mean(np.array(strap_loss),0))
    plt.plot(list(range(1, episodes + 1)),
             np.convolve(np.mean(np.array(strap_loss), 0), np.ones(100) / 100, mode='same'))
    plt.title('A2C CartPole loss, Total Runtime {}s'.format(np.round(np.sum(strap_run),3)))
    plt.show()
