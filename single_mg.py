import numpy as np
import tensorflow as tf
from env import MicrogridEnv
from env.config import MG_configs
from network import network
from network import Memory
from datetime import datetime
from FCSAC import get_market_prices
import os


class SingleMGCSACTrainer:
    def __init__(
        self,
        alpha=0.1,
        lambda_=0.1,
        constraint_threshold=0.01,
        lr_actor=5e-3,
        lr_critic=1e-2,
        lr_entropy=1e-4,
        use_priority=True,
        actor_units=(512, 256, 64, 32),
        critic_units=(128, 128, 32),
        tau=1e-3,
        gamma=0.9,
        batch_size=64,
        memory_cap=150000,
        eta_lambda=0.001,
        lambda_max=100.0,
        delta_lambda_min=-0.001,
        delta_lambda_max=0.001,
        target_entropy=-25,
        rmax=2900 * 1e-4,
        mg_id='MG1',
        episode_length=24,
        episodes=5
    ):
        self.mg_id = mg_id
        self.episode_length = episode_length
        self.episodes = episodes
        self.rmax = rmax

        self.config = self.build_config(mg_id)
        self.forecast_data = self.build_forecast(episode_length)
        self.env = MicrogridEnv(config=self.config, forecast_data=self.forecast_data, episode_length=episode_length)

        self.state_shape = self.env.observation_space.shape[0]
        self.state_low = self.env.observation_space.low.astype(np.float32)
        self.state_high = self.env.observation_space.high.astype(np.float32)

        self.action_dim_discrete = 0
        self.action_dim_continuous = self.env.action_space.shape[0]
        self.action_low = self.env.action_space.low.astype(np.float32)
        self.action_high = self.env.action_space.high.astype(np.float32)
        self.action_bound = (self.action_high - self.action_low) / 2
        self.action_shift = (self.action_high + self.action_low) / 2

        self.use_priority = use_priority
        self.memory = Memory(capacity=memory_cap) if use_priority else deque(maxlen=memory_cap)

        self.network = network(
            state_dim=self.state_shape,
            action_dim_continuous=self.action_dim_continuous,
            action_dim_discrete=self.action_dim_discrete,
            action_bound=self.action_bound,
            action_shift=self.action_shift,
            state_low=self.state_low,
            state_high=self.state_high
        )

        self.actor = self.network.actor(actor_units)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.q1_critic, self.q2_critic = self.network.critic(critic_units)
        self.q1_critic_target, self.q2_critic_target = self.network.critic(critic_units)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        self.qc_critic = self.network.constraint_critic(critic_units)
        self.qc_critic_target = self.network.constraint_critic(critic_units)
        self.qc_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)

        self.gamma = gamma
        self.tau = np.float32(tau)
        self.batch_size = batch_size
        self.alpha = tf.Variable(alpha * self.rmax, dtype=tf.float32, trainable=True)
        self.lambda_ = tf.Variable(lambda_ * self.rmax, trainable=True)
        self.constraint_threshold = constraint_threshold * self.rmax
        self.lambda_max = lambda_max * self.rmax * 1e4
        self.eta_lambda = eta_lambda * self.rmax
        self.delta_lambda_min = delta_lambda_min * self.rmax
        self.delta_lambda_max = delta_lambda_max * self.rmax
        self.target_entropy = target_entropy
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_entropy)
        self.epsilon = np.float32(1.0)

    def build_config(self, mg_id):
        config_mg = MG_configs[mg_id]
        common_keys = [
            'elec_price_buy',
            'carbon_emission_factor_grid',
            'carbon_emission_factor_gas',
            'gas_price',
            'carbon_price_buy',
            'carbon_price_sell'
        ]
        config_common = {k: MG_configs[k] for k in common_keys}
        return {**config_mg, **config_common}

    def build_forecast(self, episode_length):
        return {
            'Load_elec': np.random.rand(episode_length).tolist(),
            'Load_heat': np.random.rand(episode_length).tolist(),
            'PV': np.random.rand(episode_length).tolist(),
            'WT': np.random.rand(episode_length).tolist()
        }

    def get_action_summary(self):
        """
        收集当前 MG 一轮中的供需、电价行为等，供中心计算使用
        """
        return {
            'mg_id': self.mg_id,
            'supply': self.env.get_current_supply(),
            'demand': self.env.get_current_demand(),
            'action': self.env.last_action,
        }

    def get_market_prices():
        """
        接收协调器计算出的 P2P 电价（每小时更新）
        """
        self.env.update_p2p_prices(price_dict)

    def train(self):
        print("开始训练...（此处略去训练细节）")


if __name__ == '__main__':
    trainer = SingleMGCSACTrainer(mg_id='MG1', episode_length=24, episodes=5)
    trainer.train()
