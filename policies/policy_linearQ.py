from policies import base_policy as bp
import numpy as np
import tensorflow as tf
from time import time
import random
tf.logging.set_verbosity(tf.logging.ERROR)
from collections import deque

EPSILON = 0.05
STATE_DIM=9
BATCH_SIZE=16
DISCOUNT=0.9
LR=0.005
UPDATE_STEPS = 5
MAX_PLAYBACK=100
EXPLORATION_DECAY=0.0005
DECAY_ROUND=5000
STEPS_PER_LEARN=1
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

    return res.flatten()

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
        num_actions = len(bp.Policy.ACTIONS)
        state_dim = NUM_BOARD_VALUES*(STATE_DIM**2)
        self.W_trainable = np.random.normal(size=(num_actions, state_dim))
        self.W_periodic = np.random.normal(size=(num_actions, state_dim))
        self.playback_deque = deque([])

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        batch_size = min(BATCH_SIZE, len(self.playback_deque))
        gs_num = 0
        if batch_size > 0:
            for i in range(STEPS_PER_LEARN):
                batch_indices = random.sample(range(len(self.playback_deque)), batch_size)
                batch_arrays = np.array(self.playback_deque)[batch_indices]
                batch_prev_states = np.stack(batch_arrays[:, 0], axis=0)
                batch_prev_actions = np.stack(batch_arrays[:, 1], axis=0)
                batch_rewards = np.stack(batch_arrays[:, 2], axis=0)
                batch_next_states = np.stack(batch_arrays[:, 3], axis=0)
                gradient = np.zeros(self.W_trainable.shape)
                for j in range(batch_size):
                    future_q_val = np.max(np.dot(self.W_periodic, batch_next_states[j]))
                    last_round_trainable_out = np.dot(self.W_trainable, batch_prev_states[j])
                    delta = last_round_trainable_out[batch_prev_actions[j]] - batch_rewards[j] - DISCOUNT*future_q_val
                    gradient[batch_prev_actions[j]] += delta*self.W_trainable[batch_prev_actions[j]]
                self.W_trainable -= LR*gradient
                gs_num += 1

            if gs_num % UPDATE_STEPS == 0:
                self.W_periodic = self.W_trainable

        if round > DECAY_ROUND:
            self.epsilon -= self.exploration_decay

        if round % 100 == 0 :
            self.log("GS: %d Epsilon: %f "%(gs_num, self.epsilon))

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if prev_state is None or prev_action is None or reward is None:
            return 'F'
        new_state_vec = get_state_from_board(new_state)
        self.playback_deque.append([get_state_from_board(prev_state), get_action_index(prev_action), reward, new_state_vec])
        if len(self.playback_deque) > MAX_PLAYBACK:
            self.playback_deque.popleft()
        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)
        else:
            net_out = np.dot(self.W_periodic, new_state_vec)
            best_action_index = np.argmax(net_out)
            best_action = bp.Policy.ACTIONS[best_action_index]
            return best_action

