import os
import glob
import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class SelfPlayCallback(BaseCallback):
    def __init__(self, save_freq, model_pool_dir, verbose=1):
        super(SelfPlayCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.model_pool_dir = os.path.abspath(model_pool_dir)
        os.makedirs(self.model_pool_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_name = f"model_steps_{self.num_timesteps}.zip"
            model_path = os.path.join(self.model_pool_dir, model_name)
            
            self.model.save(model_path)
            
            if self.verbose > 0:
                print(f"\[Self-game ignition][Self-Play] New model generation: {model_name}")
            
            try:
                if hasattr(self, 'training_env') and self.training_env is not None:
                    raw_env = self.training_env.envs[0]
                    if hasattr(raw_env, 'env'): raw_env = raw_env.env
                    
                    if hasattr(raw_env, 'opponent_manager'):
                        raw_env.opponent_manager.force_load_specific_model(model_path, model_name)
            except Exception as e:
                print(f" Direct channel failure: {e}")
                
        return True


class SelfPlayCallback(BaseCallback):
    def __init__(self, save_freq, model_pool_dir, verbose=1):
        super(SelfPlayCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.model_pool_dir = model_pool_dir
        self.next_save_step = self.save_freq 
        self.version = 1

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_save_step:
            model_name = f"model_v{self.version}.zip"
            save_path = os.path.join(self.model_pool_dir, model_name)
            self.model.save(save_path)
            
            if self.verbose > 0:
                print("\n" + "="*50)
                print(f"[Checkpoint] ")
                print(f" Exact path: {save_path}")
                print("="*50 + "\n")
            
            self.version += 1
            self.next_save_step += self.save_freq
            
        return True


class SelfPlayOpponentManager:
    def __init__(self, model_pool_dir, device='cpu'):
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_pool_dir = os.path.join(current_script_dir, model_pool_dir)
        
        self.device = device
        self.current_opponent_model = None
        self.current_opponent_version = "Scripted_Zombie"
        self.model_cache = {}
        
        self.action_call_count = 0
        self.curriculum_threshold = 100000 

    def update_opponent(self):
        if not os.path.exists(self.model_pool_dir):
            print(f" [Path Warning] Folder not found: {self.model_pool_dir}")
            return 
            
        model_files = sorted(glob.glob(os.path.join(self.model_pool_dir, "*.zip")), 
                             key=os.path.getmtime)
                             
        if not model_files:
            return 
            
        if random.random() < 0.8:
            chosen_file = model_files[-1]
        else:
            chosen_file = random.choice(model_files)
            
        version_name = os.path.basename(chosen_file)
        
        if version_name == self.current_opponent_version:
            return
            
        if chosen_file not in self.model_cache:
            self.model_cache[chosen_file] = PPO.load(chosen_file, device=self.device)
            if len(self.model_cache) > 10:
                oldest_file = list(self.model_cache.keys())[0]
                del self.model_cache[oldest_file]
                
        self.current_opponent_model = self.model_cache[chosen_file]
        self.current_opponent_version = version_name
        
        print("\n" + "="*50)
        print(f"[Self-game ignition] The opponent on the left has switched network weights.")
        print(f"Exact path: {chosen_file}")
        print("="*50 + "\n")
        

    def predict_action(self, obs_10d, ball, p_left):
        if self.current_opponent_model is not None:
            action, _ = self.current_opponent_model.predict(obs_10d, deterministic=True)
            return action

        action = np.array([
            np.random.choice([0, 1, 2]),
            np.random.choice([0, 1, 2]),
            np.random.choice([0, 1, 2])
        ], dtype=np.int32)

        if ball and p_left:
            if not hasattr(self, 'last_ball_x'):
                self.last_ball_x = ball['x']
            
            dx = ball['x'] - self.last_ball_x
            
            if dx < -50.0:
                dx = 0.0
                
            self.last_ball_x = ball['x']

            if ball['x'] < 50.0 and dx >= -2.0:
                action[1] = 2
                action[2] = np.random.choice([0, 1, 2]) 
                
                if p_left['y'] > ball['y'] + 2: 
                    action[0] = 1
                elif p_left['y'] < ball['y'] - 2: 
                    action[0] = 2 
                else:
                    action[0] = 0
                    
                return action
            else:
                return action

        return np.array([0, 0, 0], dtype=np.int32)

    # def predict_action(self, obs_10d, ball, p_left):
    #     self.action_call_count += 1
        
    #     if self.current_opponent_model is not None:
    #         action, _ = self.current_opponent_model.predict(obs_10d, deterministic=True)
    #         return action

        # action = np.array([0, 0, 0], dtype=np.int32)
        # if not (ball and p_left):
        #     return action

        # if not hasattr(self, 'last_ball_x'):
        #     self.last_ball_x = ball['x']
        # dx = ball['x'] - self.last_ball_x
        # if dx < -50.0: dx = 0.0
        # self.last_ball_x = ball['x']

        # if ball['x'] < 50.0 and dx >= -2.0:
        #     action[1] = 2 
        #     action[2] = np.random.choice([0, 1, 2])
            
        #     if p_left['y'] > ball['y'] + 2: action[0] = 1 
        #     elif p_left['y'] < ball['y'] - 2: action[0] = 2 
        #     return action

        # elif ball['x'] < 84.0:
        #     if random.random() < 0.3:
        #         return np.array([np.random.choice([0, 1, 2]), 0, 0], dtype=np.int32)
        #     else:
        #         action[2] = np.random.choice([0, 1, 2]) 
                
        #         if p_left['y'] > ball['y'] + 2: action[0] = 1 
        #         elif p_left['y'] < ball['y'] - 2: action[0] = 2 
        #         return action
            
        # else:
        #     if p_left['y'] > ball['y'] + 2: action[0] = 1 
        #     elif p_left['y'] < ball['y'] - 2: action[0] = 2 
        #     return action
        
    def force_load_specific_model(self, model_path, version_name):
        try:
            print("\n" + "="*50)
            print(f"[Channel ignition] New weights received from the callback!")
            print(f"Exact path: {model_path}")
            print("="*50 + "\n")
            
            self.current_opponent_model = PPO.load(model_path, device=self.device)
            self.current_opponent_version = version_name
            self.curriculum_threshold = 0 
        except Exception as e:
            print(f"❌ [Loading failed.]: {e}")