from policies import base_policy as bp
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

EPSILON = 0.05
STATE_DIM=7
DISCOUNT=0.95
LR=0.000000001
MAX_PLAYBACK=500
DECAY_ROUND=5000
SEQUENCE_SIZE=128
# Hard coded numbers
NUM_ROTATES = {"N": 0, "E": 1, "S": 2, "W": 3}
NUM_BOARD_VALUES=11
TRAIN_DIR="train_DQN"

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


class FC2(nn.Module):
    def __init__(self):
        super(FC2, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM**2*NUM_BOARD_VALUES , 256)
        self.fc2 = nn.Linear(256 , 3)

    def forward(self, x):
        x = x.float()
        x = x.view(-1, STATE_DIM**2*NUM_BOARD_VALUES)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x)
        return x

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM**2*NUM_BOARD_VALUES , 3)

    def forward(self, x):
        x = x.float()
        x = x.view(-1, STATE_DIM**2*NUM_BOARD_VALUES)
        x = self.fc1(x)
        x = F.log_softmax(x)
        return x

class policy_grad_torch(bp.Policy):
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
        self.device = torch.device("cpu")

        self.trainable_model = FC2().to(self.device)

        pytorch_total_params = sum(p.numel() for p in self.trainable_model.parameters() if p.requires_grad)
        print("Parametes in model: ~%.1fk" % (pytorch_total_params / 1000))

        self.optimizer = torch.optim.Adam(self.trainable_model.parameters(), lr=LR)
        self.optimizer.zero_grad()

        self.playback_deque = deque([])

        self.step_number = 0
        self.action_statistics = [0,0,0]

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if len(self.playback_deque) >= SEQUENCE_SIZE:
            seq_start_idx = random.randint(0, len(self.playback_deque)-SEQUENCE_SIZE+1)
            batch_arrays = np.array(self.playback_deque)[seq_start_idx:seq_start_idx+SEQUENCE_SIZE]
            prev_states = np.stack(batch_arrays[:, 0], axis=0)
            prev_actions = np.stack(batch_arrays[:, 1], axis=0)
            rewards = np.stack(batch_arrays[:, 2], axis=0)

            discounted_rewards = np.array([rewards[t]*DISCOUNT ** t for t in range(len(rewards))])
            discounted_rewards = torch.tensor((np.cumsum(discounted_rewards[::-1])[::-1]).copy())
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)  # normalize discounted rewards
            actions_probs = torch.matmul(self.trainable_model(torch.tensor(prev_states)).view(-1,1,3).float(), torch.tensor(prev_actions).float()).view(-1,1)
            log_action_probs = torch.log(actions_probs)
            sum_log_probs =-torch.sum(log_action_probs.view(-1)*discounted_rewards)

            sum_log_probs.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        if round % 100 == 0 :
            self.log("GS: %d Epsilon: %f stats %s"%(self.step_number, self.epsilon, str(self.action_statistics/np.sum(self.action_statistics))))

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if round > DECAY_ROUND:
            self.epsilon = max(0,self.epsilon - self.exploration_decay)
        # start = time()
        if prev_state is None or prev_action is None or reward is None:
            return 'F'

        if round % 1000 == 0:
            self.action_statistics = [0, 0, 0]

        new_state_vec = get_state_from_board(new_state)
        self.playback_deque.append([get_state_from_board(prev_state), get_action_vec(prev_action), reward, new_state_vec])
        if len(self.playback_deque) > MAX_PLAYBACK:
            self.playback_deque.popleft()
        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)
        else:
            net_vals = self.trainable_model(torch.tensor([new_state_vec]))[0].detach().numpy()
            best_action_index = np.random.choice(len(bp.Policy.ACTIONS), p=net_vals)
            # best_action_index = net_vals.argmax()
            self.action_statistics[best_action_index] += 1
            best_action = bp.Policy.ACTIONS[best_action_index]
            return best_action

