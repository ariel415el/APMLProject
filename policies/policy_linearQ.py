from policies import base_policy as bp
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

EPSILON = 0.05
STATE_DIM=3
DISCOUNT=0.0
LR=0.1
DECAY_ROUND=1000
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
        return policy_args

    def init_run(self):
        self.exploration_decay = self.epsilon / (self.game_duration - self.score_scope - DECAY_ROUND)
        num_actions = len(bp.Policy.ACTIONS)
        state_dim = NUM_BOARD_VALUES*(STATE_DIM**2)
        self.W_trainable = np.random.normal(size=(num_actions, state_dim))
        self.b_trainable = np.random.normal(size=(num_actions,))

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        gs_num = 0
        new_state = get_state_from_board(new_state)
        prev_state = get_state_from_board(prev_state)
        prev_action = get_action_index(prev_action)

        future_q_val = np.max(np.dot(self.W_trainable, new_state) + self.b_trainable)
        last_round_trainable_out = np.dot(self.W_trainable, prev_state) + self.b_trainable
        delta = last_round_trainable_out[[prev_action]] - reward - DISCOUNT*future_q_val

        self.W_trainable[prev_action] -= LR*delta*prev_state
        self.b_trainable[prev_action] -= LR*delta

        gs_num += 1

        if round > DECAY_ROUND:
            self.epsilon -= self.exploration_decay

        if round % 100 == 0 :
            self.log("GS: %d Epsilon: %f"%(gs_num, self.epsilon))

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if round > DECAY_ROUND:
            self.epsilon = max(0,self.epsilon - self.exploration_decay)

        if prev_state is None or prev_action is None or reward is None:
            return 'F'

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)
        else:
            new_state_vec = get_state_from_board (new_state)
            net_out = np.dot(self.W_trainable, new_state_vec) + self.b_trainable
            best_action_index = np.argmax(net_out)
            best_action = bp.Policy.ACTIONS[best_action_index]
            return best_action

