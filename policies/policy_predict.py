from policies import base_policy as bp
import numpy as np

EPSILON = 0.05
FUTURE_DISCOUNT= 0.5

class Predict(bp.Policy):
    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        return policy_args

    def init_run(self):
        self.r_sum = 0

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum), 'VALUE')
                else:
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                self.r_sum = 0
            else:
                self.r_sum += reward

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        board, head = new_state
        head_pos, direction = head

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        else:
            act_scores = {v: 0 for v in bp.Policy.ACTIONS}
            for act in act_scores:
                next_direction = bp.Policy.TURNS[direction][act]
                next_position = head_pos.move(next_direction)
                r = next_position[0]
                c = next_position[1]
                if board[r, c] == -1:
                    act_scores[act] += 1.
                elif board[r, c] <= 5:
                    act_scores[act] -= 5.
                else:
                    act_scores[act] += 5.

                total_next_act_score = {v: 0 for v in bp.Policy.ACTIONS}
                for next_act in act_scores:
                    next_next_position = next_position.move(bp.Policy.TURNS[next_direction][next_act])
                    r = next_next_position[0]
                    c = next_next_position[1]
                    if board[r, c] == -1:
                        total_next_act_score[next_act] += 1.
                    elif board[r, c] <= 5:
                        total_next_act_score[next_act] -= 5.
                    else:
                        total_next_act_score[next_act] += 5.

                act_scores[act] += FUTURE_DISCOUNT*total_next_act_score[max(total_next_act_score, key=act_scores.get)]

            result = 'F'
            if not (act_scores['F'] == act_scores['L'] == act_scores['R']):
                max_v = max(act_scores, key=act_scores.get)
                possibles_results = [k for k,v in act_scores.items() if v == act_scores[max_v]]
                result = np.random.choice(possibles_results)

            self.log("Round %d options %s, result: %s"%(round, str(act_scores), result))
            return result

