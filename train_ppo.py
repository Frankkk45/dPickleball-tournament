import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from typing import Callable
from mlagents_envs.environment import UnityEnvironment
from stable_baselines3.common.vec_env import DummyVecEnv

from env_wrapper import PickleballEnvWrapper
from self_play_manager import SelfPlayCallback

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def make_env(executable_path, base_port, worker_id, frame_skip, agent_style):
    def _init():
        from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
        string_channel = StringSideChannel()
        channel = CustomDataChannel()
        channel.send_data(serve=212, p1=0, p2=0)

        unity_env = UnityEnvironment(
            file_name=executable_path,
            seed=1 + worker_id,        
            worker_id=worker_id,       
            base_port=base_port + worker_id, 
            side_channels=[string_channel, channel],
            # no_graphics=True,          
            additional_args=[
                '-screen-width', '640', 
                '-screen-height', '480',           
                '-vsync-count', '0',               
                '-logFile', f'worker_{worker_id}_log.txt' 
            ]
        )
        
        env = PickleballEnvWrapper(unity_env, frame_skip=frame_skip, agent_style=agent_style)
        return env
        
    return _init

def train():
    EXE_PATH = r"C:\project\dPickleball BuildFiless\dPickleball BuildFiles\Training\Windows\dp.exe"
    PORT = 5005
    FRAME_SKIP = 2
    TOTAL_TIMESTEPS = 1000000  
    SAVE_FREQ = 20000 
    MODEL_POOL_DIR = "./model_pool"
    LOG_DIR = "./ppo_logs"

    num_envs = 8 
    
    # agent style
    env_fns = [make_env(EXE_PATH, 5020, i, FRAME_SKIP, agent_style="sparse") for i in range(num_envs)]
    env = DummyVecEnv(env_fns)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 训练引擎已挂载至: {device}")

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],  # Actor (策略网络) 拓扑
            vf=[256, 256]   # Critic (价值网络/VFA) 拓扑
        ),
        activation_fn=nn.ReLU
    )

    # model = PPO(
    #     policy="MlpPolicy", 
    #     env=env,
    #     learning_rate=linear_schedule(3e-4), 
    #     n_steps=2048, 
    #     batch_size=512, 
    #     n_epochs=10,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     ent_coef=0.02,
    #     verbose=1,
    #     policy_kwargs=policy_kwargs,
    #     device=device,
    #     tensorboard_log=LOG_DIR
    # )

    existing_model_path = "./model_pool/model_v0.zip" 
    print(f"[Warm Start] Loading Agent model: {existing_model_path}")
    
    model = PPO.load(
        existing_model_path, 
        env=env, 
        custom_objects={"learning_rate": linear_schedule(3e-4)}
    )

    self_play_callback = SelfPlayCallback(
        save_freq=SAVE_FREQ, 
        model_pool_dir=MODEL_POOL_DIR,
        verbose=1
    )

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=self_play_callback)
        model.save("alphazero_pickleball_champion")
        print("[*] Training Complete")

    except KeyboardInterrupt:
        print("\n[!] Manual termination command received...")
        model.save("alphazero_interrupted_backup")
    finally:
        env.close()
        print("[*] Port release")

if __name__ == "__main__":
    train()