from policies import base_policy as bp
import numpy as np

EPSILON = 0.05

class Avoid(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        return policy_args

    def init_run(self):
        self.r_sum = 0

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # try:
        #     if round % 100 == 0:
        #         if round > self.game_duration - self.score_scope:
        #             self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum), 'VALUE')
        #         else:
        #             self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
        #         self.r_sum = 0
        #     else:
        #         self.r_sum += reward
        #
        # except Exception as e:
        #     self.log("Something Went Wrong...", 'EXCEPTION')
        #     self.log(e, 'EXCEPTION')
        return

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        board, head = new_state
        head_pos, direction = head

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        else:
            for a in list(np.random.permutation(bp.Policy.ACTIONS)):

                # get a Position object of the position in the relevant direction from the head:
                next_position = head_pos.move(bp.Policy.TURNS[direction][a])
                r = next_position[0]
                c = next_position[1]

                # look at the board in the relevant position:
                if board[r, c] > 5 or board[r, c] < 0:
                    return a

            # if all positions are bad:
            return np.random.choice(bp.Policy.ACTIONS)

