from policies import base_policy as bp
import numpy as np
import tensorflow as tf
from time import time
import random
tf.logging.set_verbosity(tf.logging.ERROR)
from collections import deque

EPSILON = 0.05
STATE_DIM=5
BATCH_SIZE=1
DISCOUNT=0.9
LR=0.025
UPDATE_STEPS = 5
MAX_PLAYBACK=100
EXPLORATION_DECAY=0.99
DECAY_ROUND=3000
NUM_ROTATES = {"N": 0, "E": 1, "S": 2, "W": 3}
NUM_BOARD_VALUES=11

def get_state_from_board(state):
    board, head = state
    head_pos, direction = head
    res = np.zeros((STATE_DIM,STATE_DIM, NUM_BOARD_VALUES))
    for r in range(STATE_DIM):
        for c in range(STATE_DIM):
            board_r = (head_pos[0]-int(STATE_DIM/2) + r) % board.shape[0]
            board_c = (head_pos[1]-int(STATE_DIM/2) + c) % board.shape[1]
            res[r, c, board[board_r, board_c] + 1] = 1

    # rotate matrix to allign all directions
    res = np.rot90(res, k=NUM_ROTATES[direction])

    return res


def get_action_vec(action):
    res = np.zeros((len(bp.Policy.ACTIONS), 1))
    res[bp.Policy.ACTIONS.index(action), 0] = 1
    return res

def get_action_index(action):
    return bp.Policy.ACTIONS.index(action)


class LinearQ(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """
    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['discount'] = float(policy_args['discount']) if 'discount' in policy_args else DISCOUNT
        policy_args['exploration_decay'] = float(policy_args['exploration_decay']) if 'exploration_decay' in policy_args else EXPLORATION_DECAY
        return policy_args

    def init_run(self):
        self.sess = tf.Session()
        # input state
        self.input_state = tf.placeholder(tf.float32, shape=(None, STATE_DIM, STATE_DIM), name='input_state')
        flatten_satate = tf.reshape(self.input_state, [-1, STATE_DIM**2])
        # define Q_hat
        self.W_periodic = tf.get_variable('W_periodic', shape=[STATE_DIM**2, len(bp.Policy.ACTIONS)], dtype=tf.float32, trainable=False, initializer=tf.contrib.layers.xavier_initializer())
        self.b_periodic = tf.get_variable('b_periodic', shape=[1, 1], dtype=tf.float32, trainable=False, initializer=tf.contrib.layers.xavier_initializer())
        self.Q_periodic_val = tf.matmul(flatten_satate,self.W_periodic) + self.b_periodic
        self.best_action_index = tf.argmax(self.Q_periodic_val, axis=1)
        self.best_action_value = tf.reduce_max(self.Q_periodic_val, reduction_indices=[1])

        # define trainable Q
        self.W_trainable = tf.get_variable('W_trainable', shape=[STATE_DIM**2, len(bp.Policy.ACTIONS)], dtype=tf.float32, trainable=True, initializer=tf.contrib.layers.xavier_initializer())
        self.b_trainable = tf.get_variable('b_trainable', shape=[1, 1], dtype=tf.float32, trainable=True, initializer=tf.contrib.layers.xavier_initializer())
        self.Q_trainable_val = tf.matmul(flatten_satate, self.W_trainable) + self.b_trainable

        # get differentiable value of Q(s,a)
        self.action_one_hot = tf.placeholder(tf.float32, shape=(None,3,1), name='action_index')
        Q_trainable_val_reshape = tf.reshape(self.Q_trainable_val, [-1,1,3])
        action_q_val = tf.matmul(Q_trainable_val_reshape, self.action_one_hot)

        # self.reward = tf.placeholder(tf.float32, shape=(None,), name='reward')
        self.target_val = tf.placeholder(tf.float32, shape=(None,), name='target_val')
        target_val = tf.reshape(self.target_val, [-1,1])
        self.loss = tf.reduce_mean((target_val - action_q_val)**2)


        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(self.loss, self.global_step)

        self.update_periodic_w = tf.assign(self.W_periodic, self.W_trainable)
        self.update_periodic_b = tf.assign(self.b_periodic, self.b_trainable)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.playback_deque = deque([])

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        batch_size = min(BATCH_SIZE, len(self.playback_deque))
        if batch_size > 0:
            batch_indices = random.sample(range(len(self.playback_deque)), batch_size)
            batch_arrays = np.array(self.playback_deque)[batch_indices]
            prev_states = np.stack(batch_arrays[:, 0], axis=0)
            prev_actions = np.stack(batch_arrays[:, 1], axis=0)
            rewards = np.stack(batch_arrays[:, 2], axis=0)
            next_states = np.stack(batch_arrays[:, 3], axis=0)
            future_q_val = self.sess.run(self.best_action_value, feed_dict={self.input_state: next_states})
            target_values = rewards + self.discount*future_q_val
            _, loss, gs_num, debug_trainable_q = self.sess.run([self.train_op, self.loss, self.global_step, self.Q_trainable_val],
                                            feed_dict={self.input_state: prev_states,
                                                       self.action_one_hot: prev_actions,
                                                       self.target_val:target_values
                                                       })


            if gs_num % UPDATE_STEPS == 0:
                self.sess.run(self.update_periodic_w)
                self.sess.run(self.update_periodic_b)
                if round > DECAY_ROUND:
                    self.epsilon *= self.exploration_decay
        if round % 100 == 0 :
            self.log("GS: %d Epsilon: %f, loss: "%(gs_num, self.epsilon)+str(loss))
            # self.log("\n"+str(prev_states))
            # self.log("\n"+str(prev_actions))
            # self.log(debug_trainable_q)
            # self.log(loss)
        # self.log("Loss: " + str(loss))

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # start = time()
        if prev_state is None or prev_action is None or reward is None:
            # print("None states encountered")
            return 'F'
        new_state_vec = get_state_from_board(new_state)
        self.playback_deque.append([get_state_from_board(prev_state), get_action_vec(prev_action), reward, new_state_vec])
        if len(self.playback_deque) > MAX_PLAYBACK:
            self.playback_deque.popleft()
        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)
        else:
            debug_all_states = self.sess.run(self.Q_periodic_val, feed_dict={self.input_state: [new_state_vec]})
            best_action_index = self.sess.run(self.best_action_index, feed_dict={self.input_state: [new_state_vec]})
            best_action = bp.Policy.ACTIONS[best_action_index[0]]
            # self.log("act took %f seconds"%(time()-start))
            return best_action

