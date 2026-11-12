import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from self_play_manager import SelfPlayOpponentManager

from hybrid_physics_engine import HybridPhysicsPredictor, State
from tracker import RobustTracker 

class PickleballEnvWrapper(gym.Env):
    def __init__(self, unity_env, frame_skip=2, max_energy=50.0, agent_style="sparse"):
        super().__init__()
        self.opponent_manager = SelfPlayOpponentManager(model_pool_dir="./model_pool", device="cpu")
        self.env = UnityParallelEnv(unity_env)
        
        self.agent_obs = self.env.possible_agents[0] 
        
        self.frame_skip = frame_skip
        self.max_energy = max_energy
        self.agent_style = agent_style
        
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        old_action_space = self.env.action_spaces[self.env.possible_agents[1]]
        self.action_space = spaces.MultiDiscrete(old_action_space.nvec)

        self.tracker = RobustTracker() 
        self.predictor = HybridPhysicsPredictor()

        self.prev_ball = None
        self.last_ball = None
        self.last_p_left = None
        self.last_p_right = None

    def reset(self, seed=None, options=None):
        obs_dict = self.env.reset() 
        obs_raw = obs_dict[self.agent_obs]['observation'][0]


        if random.random() < 0.5:
            self.agent = self.env.possible_agents[1]       
            self.agent_other = self.env.possible_agents[0] 
            self.am_i_right = True
        else:
            self.agent = self.env.possible_agents[0]       
            self.agent_other = self.env.possible_agents[1] 
            self.am_i_right = False


        # random_k = np.clip(np.random.normal(0.1, 0.02), 0.0, 0.3)
        # random_omega = np.random.uniform(-2.0, 2.0) 
        # random_restitution = np.random.uniform(0.7, 0.9)
        
        # self.predictor.update_physics_parameters(random_k, random_omega, random_restitution)
        p_left, p_right, ball, court_bounds, _ = self.tracker.track(obs_raw)
        
        self.prev_ball = ball 
        self.last_ball = ball
        self.last_p_left = p_left
        self.last_p_right = p_right
        
        if self.am_i_right:
            initial_obs = self._build_10d_observation(ball, p_right, p_left, target_x=168.0)
        else:
            initial_obs = self._build_10d_mirrored(ball, p_left, p_right, target_x=0.0)
        
        if hasattr(self, 'opponent_manager') and self.opponent_manager is not None:
            self.opponent_manager.update_opponent()
            
        return initial_obs, {}

    def _build_10d_observation(self, ball, p_right, p_left, target_x=168.0, override_phys=None):
        if override_phys is not None:
            y_pred, t_arrive, is_contact, energy = override_phys
        else:
            y_pred, t_arrive, is_contact, energy = self.predictor.step_and_predict(
                ball, self.prev_ball, [self.last_p_left, p_right], target_x
            )
            
        px = p_right['x'] / 168.0 if p_right else 0.8
        py = p_right['y'] / 84.0 if p_right else 0.5
        bx = ball['x'] / 168.0 if ball else 0.5
        by = ball['y'] / 84.0 if ball else 0.5
        
        opp_x = p_left['x'] / 168.0 if p_left else 0.2
        opp_y = p_left['y'] / 84.0 if p_left else 0.5
        
        energy_norm = np.clip(energy / self.max_energy, 0.0, 1.0)
        
        return np.array([px, py, bx, by, y_pred, t_arrive, is_contact, energy_norm, opp_x, opp_y], dtype=np.float32)

    def _build_10d_mirrored(self, ball, p_left, p_right, target_x=0.0, override_phys=None):
        if override_phys is not None:
            y_pred, t_arrive, is_contact, energy = override_phys
        else:
            y_pred, t_arrive, is_contact, energy = self.predictor.step_and_predict(
                ball, self.prev_ball, [self.last_p_right, p_left], target_x
            )
            
        px_mirrored = 1.0 - (p_left['x'] / 168.0) if p_left else 0.8
        py = p_left['y'] / 84.0 if p_left else 0.5
        
        bx_mirrored = 1.0 - (ball['x'] / 168.0) if ball else 0.5
        by = ball['y'] / 84.0 if ball else 0.5
        
        opp_x_mirrored = 1.0 - (p_right['x'] / 168.0) if p_right else 0.2
        opp_y = p_right['y'] / 84.0 if p_right else 0.5
        
        energy_norm = np.clip(energy / self.max_energy, 0.0, 1.0)
        
        return np.array([
            px_mirrored, py, 
            bx_mirrored, by, 
            y_pred, t_arrive, is_contact, energy_norm, 
            opp_x_mirrored, opp_y
        ], dtype=np.float32)

    def _get_scripted_action(self, ball, p_left):
        action = np.array([0, 1, 0], dtype=np.int32)
        if ball and p_left:
            if p_left['y'] > ball['y'] + 5: action[0] = 1
            elif p_left['y'] < ball['y'] - 5: action[0] = 2
            action[2] = np.random.choice([0, 1, 2])
        return action

    def _get_scripted_action(self, ball, p_left):
        action = np.array([0, 1, 0], dtype=np.int32) 
        
        if ball and p_left:
            if p_left['y'] > ball['y'] + 2: 
                action[0] = 1
            elif p_left['y'] < ball['y'] - 2: 
                action[0] = 2
                
            if ball['x'] < 84.0:
                action[1] = 2 
                action[2] = np.random.choice([0, 1, 2])
            else:
                action[1] = 1 
                
        return action

    def step(self, action):
        if self.am_i_right:
            y_pred_opp, t_arrive_opp, is_contact_opp, energy_opp = self.predictor.step_and_predict(
                self.last_ball, self.prev_ball, [self.last_p_right, self.last_p_left], target_x=168.0
            )
            opp_obs_10d = self._build_10d_mirrored(
                self.last_ball, self.last_p_left, self.last_p_right, 
                target_x=168.0, 
                override_phys=(y_pred_opp, t_arrive_opp, is_contact_opp, energy_opp)
            )
        else:
            y_pred_opp, t_arrive_opp, is_contact_opp, energy_opp = self.predictor.step_and_predict(
                self.last_ball, self.prev_ball, [self.last_p_left, self.last_p_right], target_x=0.0
            )
            opp_obs_10d = self._build_10d_observation(
                self.last_ball, self.last_p_right, self.last_p_left, 
                target_x=0.0,
                override_phys=(y_pred_opp, t_arrive_opp, is_contact_opp, energy_opp)
            )

        opp_action = self.opponent_manager.predict_action(opp_obs_10d, self.last_ball, self.last_p_left)

        # opp_action = self.opponent_manager.predict_action(None, self.last_ball, self.last_p_left)
        
        agent_action = np.copy(action)
        env_opp_action = np.copy(opp_action)
        
        if self.am_i_right: 
            if env_opp_action[1] == 1: env_opp_action[1] = 2
            elif env_opp_action[1] == 2: env_opp_action[1] = 1
        else:
            if agent_action[1] == 1: agent_action[1] = 2
            elif agent_action[1] == 2: agent_action[1] = 1

        actions = {self.agent: agent_action, self.agent_other: env_opp_action}

        macro_env_reward = 0.0
        terminations_dict = {self.agent: False}
        infos_dict = {self.agent: {}}
        obs_raw = None

        for _ in range(self.frame_skip):
            obs_dict, rewards, terminations, infos = self.env.step(actions)
            macro_env_reward += (rewards[self.agent] - rewards[self.agent_other])
            terminations_dict = terminations
            infos_dict = infos
            obs_raw = obs_dict[self.agent_obs]['observation'][0]
            if terminations[self.agent]:
                break

        p_left, p_right, ball, _, _ = self.tracker.track(obs_raw)

        if self.am_i_right:
            obs_10d = self._build_10d_observation(ball, p_right, p_left, target_x=168.0)
        else:
            obs_10d = self._build_10d_mirrored(ball, p_left, p_right, target_x=0.0)

        total_reward = 0.0
        # if ball and ball['x'] > 84.0:
        #     shaped_reward += self.time_penalty

        # prev_state = self.predictor.current_state
        # prev_energy = self.predictor.accumulated_energy

        # obs_10d = self._build_10d_observation(ball, p_right, p_left)
        
        # current_state = self.predictor.current_state
        # if prev_state == State.Q3_PADDLE_CONTACT and current_state == State.Q1_FREE_FLIGHT:
        #     if ball and self.last_ball and (ball['x'] - self.last_ball['x'] < 0):
        #         kinetic_bonus = prev_energy * self.kinetic_reward_multiplier
        #         shaped_reward += kinetic_bonus

        # self.last_ball = ball
        # self.last_p_left = p_left
        # self.last_p_right = p_right

        # total_reward = macro_env_reward + shaped_reward
        # ==========================================
        is_win = (macro_env_reward > 0.5)
        is_loss = (macro_env_reward < -0.5)
        
        
        if is_win or is_loss:
            if self.agent_style == "conservative":
                total_reward = 0.0 if is_win else -2.0
            elif self.agent_style == "aggressive":
                total_reward = 2.0 if is_win else 0.0
            elif self.agent_style == "sparse":
                total_reward = 1.0 if is_win else -1.0
            
        self.last_ball = ball
        self.last_p_left = p_left
        self.last_p_right = p_right

        if macro_env_reward != 0:
            if hasattr(self, 'opponent_manager') and self.opponent_manager is not None:
                self.opponent_manager.update_opponent()

        return obs_10d, total_reward, terminations_dict[self.agent], False, infos_dict[self.agent]