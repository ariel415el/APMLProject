from policies import base_policy as bp
import numpy as np
import tensorflow as tf
from time import time
import random
tf.logging.set_verbosity(tf.logging.ERROR)
from collections import deque
from math import ceil
import os
# Tunable parameters

EPSILON = 0.05
STATE_DIM=7
BATCH_SIZE=64
DISCOUNT=0.99
LR=0.00001
UPDATE_STEPS = 10
MAX_PLAYBACK=500
DECAY_ROUND=5000

# Hard coded numbers
NUM_ROTATES = {"N": 0, "E": 1, "S": 2, "W": 3}
NUM_BOARD_VALUES=11
TRAIN_DIR="train_DQN"

def get_test_states():
    """" Creates hand crafted states to evaluate the progress of the learner on"""
    pos_states = []
    neg_states = []
    middle = int(STATE_DIM/2)
    default_state = np.zeros((STATE_DIM, STATE_DIM, NUM_BOARD_VALUES))
    default_state[:,:,0] = 1
    for k in range(middle, STATE_DIM):
        default_state[k, middle, 1] = 1

    # states += [default_state]

    close_poses = [(middle-1,middle-1), (middle,middle-1), (middle-1,middle), (middle+1,middle+1), (middle+1,middle-1), (middle-1,middle+1), (middle,middle+1)]
    far_poses = [(middle-2,middle-2), (middle,middle-2), (middle-2,middle), (middle+2,middle+2), (middle+2,middle-2), (middle-2,middle+2), (middle,middle+2)]
    # random.shuffle(far_poses)
    # random.shuffle(close_poses)
    for p1 in close_poses+far_poses:
        s = default_state.copy()
        s[p1[0], p1[0], 8] = 1
        pos_states += [s]

    for p1 in close_poses+far_poses:
        s = default_state.copy()
        s[p1[0], p1[0], 6] = 1
        neg_states += [s]

    return pos_states, neg_states

def get_state_from_board(state, get_pos=False):
    """"Convert the full game state to a meaningfull represantation of it """
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
    if get_pos:
        return res, head_pos
    else:
        return res


def get_action_vec(action):
    """ Convert an action string to its one hot equivalent """
    res = np.zeros((len(bp.Policy.ACTIONS), 1))
    res[bp.Policy.ACTIONS.index(action), 0] = 1
    return res

def get_action_index(action):
    return bp.Policy.ACTIONS.index(action)


class MLP(object):
    def __init__(self, model_instance_name, trainable):
        num_actions = len(bp.Policy.ACTIONS)
        hidden_layer = 128
        with tf.variable_scope(model_instance_name):
            self.input_state = tf.placeholder(tf.float32, shape=(None, STATE_DIM, STATE_DIM, NUM_BOARD_VALUES),name='input_state')
            self.W_1 = tf.get_variable('W_1', shape=[NUM_BOARD_VALUES*STATE_DIM**2, hidden_layer], dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            self.b_1 = tf.get_variable('b_1', shape=[hidden_layer, ], dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            self.W_2 = tf.get_variable('W_2', shape=[hidden_layer, num_actions], dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            self.b_2 = tf.get_variable('b_2', shape=[num_actions, ], dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            flatten_satate = tf.reshape(self.input_state, [-1, NUM_BOARD_VALUES * STATE_DIM ** 2])
            feature_map = tf.matmul(flatten_satate, self.W_1) + self.b_1
            feature_map = tf.nn.relu(feature_map)
            self.probs = tf.matmul(feature_map, self.W_2) + self.b_2

            self.best_action_index = tf.argmax(self.probs, axis=1)
            self.best_action_value = tf.reduce_max(self.probs, reduction_indices=[1])

    def get_net_out(self):
        return self.probs

    def __call__(self, sess, state):
        net_out = sess.run(self.probs, feed_dict={self.input_state:state})
        return net_out

    def get_best_action_value(self, sess, state):
        return sess.run(self.best_action_value, feed_dict={self.input_state:state})

    def get_best_action_index(self, sess, state):
        return sess.run(self.best_action_index, feed_dict={self.input_state:state})

    def get_variables(self):
        return [self.W_1, self.b_1, self.W_2, self.b_2]


class linear_model(object):
    def __init__(self, model_instance_name, trainable):
        num_actions = len(bp.Policy.ACTIONS)
        with tf.variable_scope(model_instance_name):
            self.input_state = tf.placeholder(tf.float32, shape=(None, STATE_DIM, STATE_DIM, NUM_BOARD_VALUES),name='input_state')
            self.W = tf.get_variable('W', shape=[NUM_BOARD_VALUES*STATE_DIM**2, num_actions], dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable('b', shape=[num_actions, ], dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            flatten_satate = tf.reshape(self.input_state, [-1, NUM_BOARD_VALUES * STATE_DIM ** 2])

            self.probs = tf.matmul(flatten_satate, self.W) + self.b

            self.best_action_index = tf.argmax(self.probs, axis=1)
            self.best_action_value = tf.reduce_max(self.probs, reduction_indices=[1])

    def get_net_out(self):
        return self.probs

    def __call__(self, sess, state):
        net_out = sess.run(self.probs, feed_dict={self.input_state:state})
        return net_out

    def get_best_action_value(self, sess, state):
        return sess.run(self.best_action_value, feed_dict={self.input_state:state})

    def get_best_action_index(self, sess, state):
        return sess.run(self.best_action_index, feed_dict={self.input_state:state})

    def get_variables(self):
        return [self.W, self.b]


class small_conv_model(object):
    def __init__(self, model_instance_name, trainable):
        num_actions = len(bp.Policy.ACTIONS)
        fmaps_channels= [1]
        hidden_fc_dim=fmaps_channels[0]*STATE_DIM ** 2
        with tf.variable_scope(model_instance_name):
            self.input_state = tf.placeholder(tf.float32, shape=(None, STATE_DIM, STATE_DIM, NUM_BOARD_VALUES),name='input_state')

            self.conv_1_w = tf.get_variable('conv_1_w', shape=[1, 1, 11, fmaps_channels[0]], dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            self.conv_b_1 = tf.get_variable('conv_b_1', shape=[fmaps_channels[0]], dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            self.W_2 = tf.get_variable('W_2', shape=[hidden_fc_dim, num_actions], dtype=tf.float32, trainable=trainable,initializer=tf.contrib.layers.xavier_initializer())
            self.b_2 = tf.get_variable('b_2', shape=[num_actions,], dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            feature_map = tf.nn.conv2d(self.input_state, self.conv_1_w, strides=[1, 1, 1, 1], padding='SAME')
            feature_map = tf.nn.bias_add(feature_map, self.conv_b_1)
            feature_map = tf.nn.relu(feature_map)

            flatten_features = tf.reshape(feature_map, [-1, fmaps_channels[0] * STATE_DIM ** 2])

            self.probs = tf.matmul(flatten_features, self.W_2) + self.b_2

            self.best_action_index = tf.argmax(self.probs, axis=1)
            self.best_action_value = tf.reduce_max(self.probs, reduction_indices=[1])

    def get_net_out(self):
        return self.probs

    def __call__(self, sess, state):
        net_out = sess.run(self.probs, feed_dict={self.input_state:state})
        return net_out

    def get_best_action_value(self, sess, state):
        return sess.run(self.best_action_value, feed_dict={self.input_state:state})

    def get_best_action_index(self, sess, state):
        return sess.run(self.best_action_index, feed_dict={self.input_state:state})

    def get_variables(self):
        return [self.conv_1_w, self.conv_b_1, self.W_2, self.b_2]


class conv_model(object):
    def __init__(self, model_instance_name, trainable):
        num_actions = len(bp.Policy.ACTIONS)
        fmaps_channels= [4,2]
        hidden_fc_dim=8
        with tf.variable_scope(model_instance_name):
            self.input_state = tf.placeholder(tf.float32, shape=(None, STATE_DIM, STATE_DIM, NUM_BOARD_VALUES),name='input_state')
            # self.conv_weights = []
            # for i in range(1, len(fmaps_channels)):
            #     self.conv_weights += [tf.get_variable('conv_%d_w'%i, shape=[3, 3, fmaps_channels[i-1], fmaps_channels[i]], dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())]
            #     self.conv_weights += [tf.get_variable('conv_%d_b'%i, shape=[fmaps_channels[i]], dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())]

            self.conv_1_w = tf.get_variable('conv_1_w', shape=[3, 3, NUM_BOARD_VALUES, fmaps_channels[0]], dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            self.conv_1_b = tf.get_variable('conv_1_b', shape=[fmaps_channels[0]], dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            self.conv_2_w = tf.get_variable('conv_2_w', shape=[3, 3, fmaps_channels[0], fmaps_channels[1]], dtype=tf.float32, trainable=True, initializer=tf.contrib.layers.xavier_initializer())
            self.conv_2_b = tf.get_variable('conv_2_b', shape=[fmaps_channels[1]], dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            # self.conv_3_w = tf.get_variable('conv_3_w', shape=[3, 3, fmaps_channels[1], fmaps_channels[2]], dtype=tf.float32, trainable=True, initializer=tf.contrib.layers.xavier_initializer())
            # self.conv_3_b = tf.get_variable('conv_3_b', shape=[fmaps_channels[2]], dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            self.W_1 = tf.get_variable('W_1', shape=[fmaps_channels[-1]*STATE_DIM **2, hidden_fc_dim], dtype=tf.float32, trainable=trainable,initializer=tf.contrib.layers.xavier_initializer())
            self.b_1 = tf.get_variable('b_1', shape=[hidden_fc_dim,], dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            self.W_2 = tf.get_variable('W_2', shape=[hidden_fc_dim, num_actions], dtype=tf.float32, trainable=trainable,initializer=tf.contrib.layers.xavier_initializer())
            self.b_2 = tf.get_variable('b_2', shape=[num_actions,], dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            feature_map = tf.nn.conv2d(self.input_state, self.conv_1_w, strides=[1, 1, 1, 1], padding='SAME')
            feature_map = tf.nn.bias_add(feature_map, self.conv_1_b)
            feature_map = tf.nn.relu(feature_map)

            feature_map = tf.nn.conv2d(feature_map, self.conv_2_w, strides=[1, 1, 1, 1], padding='SAME')
            feature_map = tf.nn.bias_add(feature_map, self.conv_2_b)
            feature_map = tf.nn.relu(feature_map)

            # feature_map = tf.nn.conv2d(feature_map, self.conv_3_w, strides=[1, 1, 1, 1], padding='SAME')
            # feature_map = tf.nn.bias_add(feature_map, self.conv_3_b)
            # feature_map = tf.nn.relu(feature_map)

            flatten_features = tf.reshape(feature_map, [-1, fmaps_channels[-1] * STATE_DIM ** 2])
            flatten_features = tf.matmul(flatten_features, self.W_1) + self.b_1

            flatten_features = tf.nn.relu(flatten_features)

            self.probs = tf.matmul(flatten_features, self.W_2) + self.b_2

            self.best_action_index = tf.argmax(self.probs, axis=1)
            self.best_action_value = tf.reduce_max(self.probs, reduction_indices=[1])

    def get_net_out(self):
        return self.probs

    def __call__(self, sess, state):
        net_out = sess.run(self.probs, feed_dict={self.input_state:state})
        return net_out

    def get_best_action_value(self, sess, state):
        return sess.run(self.best_action_value, feed_dict={self.input_state:state})

    def get_best_action_index(self, sess, state):
        return sess.run(self.best_action_index, feed_dict={self.input_state:state})

    def get_variables(self):
        # return [self.conv_1_w, self.conv_1_b, self.conv_2_w, self.conv_2_b, self.conv_3_w, self.conv_3_b, self.W_1, self.b_1, self.W_2, self.b_2]
        return [self.conv_1_w, self.conv_1_b, self.conv_2_w, self.conv_2_b, self.W_1, self.b_1, self.W_2, self.b_2]


class DQN(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """
    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['discount'] = float(policy_args['discount']) if 'discount' in policy_args else DISCOUNT
        return policy_args

    def init_run(self):
        self.sess = tf.Session()

        # define Q function
        self.peroodic_model = MLP("Periodic", trainable=False)
        self.trainable_model = MLP("Trainable", trainable=True)

        # get differentiable value of Q(s,a)
        self.action_one_hot = tf.placeholder(tf.float32, shape=(None,3,1), name='action_index')
        self.Q_trainable_val = self.trainable_model.get_net_out()
        Q_trainable_val_reshape = tf.reshape(self.Q_trainable_val, [-1,1,3])
        action_q_val = tf.matmul(Q_trainable_val_reshape, self.action_one_hot)
        self.target_val = tf.placeholder(tf.float32, shape=(None,), name='target_val')
        target_val = tf.reshape(self.target_val, [-1,1])
        self.loss = tf.reduce_mean((target_val - action_q_val)**2)

        # Define optimizer
        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(self.loss, self.global_step)

        # define periodic update operation
        trainable_vars = self.trainable_model.get_variables()
        periodic_vars = self.peroodic_model.get_variables()
        self.assign_ops = [tf.assign(periodic_vars[i], trainable_vars[i]) for i in range(len(periodic_vars))]

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.exploration_decay = self.epsilon / (self.game_duration - self.score_scope - DECAY_ROUND)
        self.batch_size = BATCH_SIZE
        # input state

        self.playback_deque = deque([])

        os.makedirs(TRAIN_DIR, exist_ok=True)
        self.loss_summary = tf.summary.scalar("Train loss", self.loss)
        self.train_writer = tf.summary.FileWriter(TRAIN_DIR, self.sess.graph_def)

        self.action_statistics = [0,0,0]
        self.pos_states, self.neg_states = get_test_states()

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if too_slow:
            self.batch_size = max(4, self.batch_size - 4)
        batch_size = min(self.batch_size, len(self.playback_deque))
        if batch_size > 0:
            batch_indices = random.sample(range(len(self.playback_deque)), batch_size)
            batch_arrays = np.array(self.playback_deque)[batch_indices]
            prev_states = np.stack(batch_arrays[:, 0], axis=0)
            prev_actions = np.stack(batch_arrays[:, 1], axis=0)
            rewards = np.stack(batch_arrays[:, 2], axis=0)
            next_states = np.stack(batch_arrays[:, 3], axis=0)

            future_q_val = self.peroodic_model.get_best_action_value(self.sess, next_states)

            target_values = rewards + self.discount*future_q_val
            _, loss, gs_num, loss_summary = self.sess.run([self.train_op, self.loss, self.global_step, self.loss_summary],
                                            feed_dict={self.trainable_model.input_state: prev_states,
                                                       self.action_one_hot: prev_actions,
                                                       self.target_val: target_values
                                                       })

            self.train_writer.add_summary(loss_summary, gs_num)

            if gs_num % UPDATE_STEPS == 0:
                self.sess.run(self.assign_ops)

            if round % 1000 == 0:
                self.action_statistics = [0, 0, 0]

            if round % 100 == 0 :
                pos_vals = self.peroodic_model.get_best_action_value(self.sess, self.pos_states)
                neg_vals = self.peroodic_model.get_best_action_value(self.sess, self.neg_states)
                self.log("GS: %d Epsilon: %f, bs: %d statistics: %s, test_pos: %f, test_neg: %f"%(gs_num, self.epsilon, self.batch_size,
                                                                               str(self.action_statistics / np.sum(self.action_statistics)), pos_vals.mean(), neg_vals.mean()))

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if round > DECAY_ROUND:
            self.epsilon = max(0,self.epsilon - self.exploration_decay)
        if prev_state is None or prev_action is None or reward is None:
            random_action_index = random.randint(0,2)
            self.action_statistics[random_action_index] += 1
            return bp.Policy.ACTIONS[random_action_index]

        new_state_vec, head_pos = get_state_from_board(new_state, get_pos=True)
        self.playback_deque.append([get_state_from_board(prev_state), get_action_vec(prev_action), reward, new_state_vec])
        if len(self.playback_deque) > MAX_PLAYBACK:
            self.playback_deque.popleft()

        if np.random.rand() < self.epsilon:
            random_action_index = random.randint(0,2)
            self.action_statistics[random_action_index] += 1
            return bp.Policy.ACTIONS[random_action_index]
        else:
            best_action_index = self.peroodic_model.get_best_action_index(self.sess, [new_state_vec])
            best_action = bp.Policy.ACTIONS[best_action_index[0]]
            self.action_statistics[best_action_index[0]] += 1
            return best_action

