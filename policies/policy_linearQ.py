from policies import base_policy as bp
import numpy as np
import tensorflow as tf
from time import time

EPSILON = 0.05
STATE_DIM=5
BATCH_SIZE=1
DISCOUNT=0.5
LR=0.001
def get_state_from_board(state):
    board, head = state
    head_pos, direction = head
    res = np.zeros((STATE_DIM,STATE_DIM))
    for r in range(STATE_DIM):
        for c in range(STATE_DIM):
            board_r = (head_pos[0]-2 + r) % board.shape[0]
            board_c = (head_pos[1]-2 + r) % board.shape[1]
            res[r, c] = board[board_r, board_c]

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
        return policy_args

    def init_run(self):
        self.sess = tf.Session()
        self.W = tf.get_variable('W', shape=[STATE_DIM**2, len(bp.Policy.ACTIONS)], dtype=tf.float32, trainable=True, initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable('b', shape=[1, 1], dtype=tf.float32, trainable=True, initializer=tf.contrib.layers.xavier_initializer())
        self.input_state = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 5, 5), name='input_state')
        s_f = tf.reshape(self.input_state, [-1, STATE_DIM**2])
        self.Q_val = tf.matmul(s_f, self.W) + self.b
        self.best_action_index = tf.argmax(self.Q_val, axis=1)
        self.best_action_value = tf.reduce_max(self.Q_val, reduction_indices=[1])


        self.target_val = tf.placeholder(tf.float32, shape=(BATCH_SIZE,1), name='target_val')
        self.action_one_hot = tf.placeholder(tf.float32, shape=(BATCH_SIZE,3,1), name='action_index')
        b_q_val = tf.reshape(self.Q_val, [-1,1,3])
        action_q_val = tf.matmul(b_q_val, self.action_one_hot)
        self.loss = (self.target_val - action_q_val)**2

        # self.max = tf.reduce_max(self.net_out, reduction_indices=[1])

        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(self.loss, self.global_step)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        prev_state_vec = get_state_from_board(prev_state)
        new_state_vec = get_state_from_board(new_state)
        prev_action_vec = get_action_vec(prev_action)
        future_q_val = self.sess.run(self.best_action_value,feed_dict={self.input_state:[new_state_vec]})
        target_value = reward + self.discount*future_q_val
        _, loss, gs_num = self.sess.run([self.train_op, self.loss, self.global_step],
                                        feed_dict={self.input_state:[prev_state_vec],
                                                   self.target_val:[target_value],
                                                   self.action_one_hot:[prev_action_vec]})
        # self.log("Loss: " + str(loss))

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)
        else:
            start = time()
            state = get_state_from_board(new_state)

            best_action_index = self.sess.run(self.best_action_index, feed_dict={self.input_state: [state]})
            best_action = bp.Policy.ACTIONS[best_action_index[0]]
            # self.log("act took %f seconds"%(time()-start))
            return best_action

