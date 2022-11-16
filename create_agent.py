import gym

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, EpisodicLifeEnv, MaxAndSkipEnv, NoopResetEnv
# from stable_baselines3.common.
from stable_baselines3.common.vec_env import VecFrameStack
import wandb
from wandb.integration.sb3 import WandbCallback

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


if __name__ == '__main__':
    wandb.init(project='PPO')
    # Parallel environments
    def wrap_env(env):
        env = TimeLimit(env, 100_000)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        env = ClipRewardEnv(env)
        

        return env

    env = make_vec_env("QbertNoFrameskip-v4", n_envs=12, env_kwargs={'obs_type': 'ram'}, wrapper_class=wrap_env)
    env = VecFrameStack(env, 4)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='/data1/s3092593/qbert_ppo')
    model.learn(total_timesteps=1_000_000_000, callback=WandbCallback())
    model.save("ppo_qbert")








    