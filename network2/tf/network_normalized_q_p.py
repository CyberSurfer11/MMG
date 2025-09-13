import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, LeakyReLU, BatchNormalization, Dropout, LayerNormalization

tf.keras.backend.set_floatx('float32')

# 输入的状态是原始的，进行归一化状态。得到的动作是归一化的
class network:
    def __init__(self, state_dim, action_dim_continuous, action_dim_discrete,
                 action_bound, action_shift, state_low, state_high):
        """
        初始化网络
        :param state_dim: 状态维度
        :param action_dim_continuous: 连续动作维度
        :param action_dim_discrete: 离散动作维度
        :param action_bound: 连续动作的范围
        :param action_shift: 连续动作的偏移量
        :param state_low: 状态的下界，列表或数组
        :param state_high: 状态的上界，列表或数组
        """
        self.state_dim = state_dim
        self.action_dim_continuous = action_dim_continuous
        # 如需启用离散动作，请把下一行改回：self.action_dim_discrete = action_dim_discrete
        self.action_dim_discrete = 0
        self.action_bound = action_bound
        self.action_shift = action_shift

        # 状态的上下限
        self.state_low = np.array(state_low, dtype=np.float32)
        self.state_high = np.array(state_high, dtype=np.float32)
        self.state_range = self.state_high - self.state_low

    # ---------------- Actor ----------------
    # 输入原始状态；输出[mu, sigma, sampled_cont_action]，若有离散则再加[ sampled_disc_action, logits ]
    # 最小改动：仅对隐藏层命名做“浅层/深层”的标记，不改变结构和输出
    def actor(self, units=(512, 256, 64, 32), tau_=0.5):
        state_input = Input(shape=(self.state_dim,), name="State_Input")

        # 归一化状态
        normalized_state = Lambda(
            lambda x: (x - self.state_low) / self.state_range,
            name="State_Normalization"
        )(state_input)

        # 划分浅层/深层的分界索引（不新增参数，按层数一半划分；至少保留1层为浅层）
        total_layers = len(units)
        shallow_count = max(1, total_layers // 2)

        # 第1层（浅层）
        x = Dense(units[0], kernel_initializer=tf.keras.initializers.he_uniform(),
                  name="Actor_Shared_L0")(normalized_state)
        x = LeakyReLU(name="Actor_Shared_L0_Act")(x)

        # 其余层（根据索引标注为浅层/深层）
        for index in range(1, total_layers):
            if index < shallow_count:
                layer_name = f"Actor_Shared_L{index}"
                act_name = f"Actor_Shared_L{index}_Act"
            else:
                deep_idx = index - shallow_count
                layer_name = f"Actor_Deep_L{deep_idx}"
                act_name = f"Actor_Deep_L{deep_idx}_Act"

            x = Dense(units[index], kernel_initializer=tf.keras.initializers.he_uniform(),
                      name=layer_name)(x)
            x = LeakyReLU(name=act_name)(x)

        # 连续动作：均值与标准差
        mu_output = Dense(self.action_dim_continuous,
                          kernel_initializer=tf.keras.initializers.he_uniform(),
                          name="Cont_Output_Mean")(x)
        mu_output = Lambda(lambda t: tf.tanh(t), name="Cont_Output_Mean_Tanh")(mu_output)  # 保持[-1,1]

        sigma_output = Dense(self.action_dim_continuous, activation="softplus",
                             kernel_initializer=tf.keras.initializers.he_uniform(),
                             name="Cont_Output_StdDev")(x)
        sigma_output = Lambda(lambda t: tf.clip_by_value(t, 0.1, np.inf),
                              name="Cont_Output_StdDev_Clip")(sigma_output)

        # 重参数化采样 + tanh
        epsilon = Lambda(lambda t: tf.random.normal(shape=tf.shape(t)),
                         name="Cont_Output_Eps")(sigma_output)
        sampled_cont_action = Lambda(lambda t: t[0] + t[1] * t[2],
                                     name="Cont_Output_Reparam")([mu_output, sigma_output, epsilon])
        sampled_cont_action = Lambda(lambda t: tf.tanh(t),
                                     name="Cont_Output_Tanh")(sampled_cont_action)

        # 离散动作（保持原有判断与输出不变）
        if self.action_dim_discrete > 0:
            logits = Dense(self.action_dim_discrete,
                           kernel_initializer=tf.keras.initializers.he_uniform(),
                           name="Disc_Output_Logits")(x)

            def gumbel_softmax_sample(logits, tau_=0.5):
                u = tf.random.uniform(tf.shape(logits), minval=1e-8, maxval=1)
                g = -tf.math.log(-tf.math.log(u))
                return tf.nn.softmax((logits + g) / tau_)

            sampled_disc_action = Lambda(lambda t: gumbel_softmax_sample(t, tau_),
                                         name="Disc_Output_Gumbel")(logits)

            return Model(state_input,
                         [mu_output, sigma_output, sampled_cont_action, sampled_disc_action, logits])

        return Model(state_input, [mu_output, sigma_output, sampled_cont_action])

    # ---------------- Critic（双Q）----------------
    # 最小改动：在“共享主干”内部用命名标注浅层/深层；输入输出保持不变
    def critic(self, units=(128, 128, 32)):
        """
        共享主干 + 两个输出 head 的 Critic 网络
        输入：S, A_cont（若有则还包含 A_disc）
        输出：[Q1, Q2]
        """
        # 输入
        state_input = Input(shape=(self.state_dim,), name="State_Input")
        cont_action_input = Input(shape=(self.action_dim_continuous,), name="Cont_Action_Input")

        # 状态归一化
        normalized_state = Lambda(
            lambda x: (x - self.state_low) / self.state_range,
            name="State_Normalization"
        )(state_input)

        if self.action_dim_discrete > 0:
            disc_action_input = Input(shape=(self.action_dim_discrete,), name="Disc_Action_Input")
            concat = Concatenate(axis=-1, name="Concat_Input")([normalized_state, cont_action_input, disc_action_input])
            inputs = [state_input, cont_action_input, disc_action_input]
        else:
            concat = Concatenate(axis=-1, name="Concat_Input")([normalized_state, cont_action_input])
            inputs = [state_input, cont_action_input]

        # 在主干里按一半划分“浅层/深层”（最小改动，仅改命名）
        total_layers = len(units)
        shallow_count = max(1, total_layers // 2)

        # 第1个主干层（浅层）
        x = Dense(units[0], kernel_initializer=tf.keras.initializers.he_uniform(),
                  name="Critic_Shared_L0")(concat)
        x = LeakyReLU(name="Critic_Shared_L0_Act")(x)
        x = LayerNormalization(name="Critic_Shared_LayerNorm0")(x)

        # 其余主干层
        for index in range(1, total_layers):
            if index < shallow_count:
                lname = f"Critic_Shared_L{index}"
                aname = f"Critic_Shared_L{index}_Act"
                dname = f"Critic_Shared_Dropout{index}"
            else:
                deep_idx = index - shallow_count
                lname = f"Critic_Deep_L{deep_idx}"
                aname = f"Critic_Deep_L{deep_idx}_Act"
                dname = f"Critic_Deep_Dropout{deep_idx}"

            x = Dense(units[index], kernel_initializer=tf.keras.initializers.he_uniform(), name=lname)(x)
            x = LeakyReLU(name=aname)(x)
            x = Dropout(0.1, name=dname)(x)

        # 双Q输出
        q1_output = Dense(1, activation="linear", name="Q1_Output")(x)
        q2_output = Dense(1, activation="linear", name="Q2_Output")(x)

        return Model(inputs=inputs, outputs=[q1_output, q2_output])

    # ---------------- 约束 Critic（保持原样）----------------
    def constraint_critic(self, units=(128, 128, 32)):
        """ 单独的约束 Critic Q_c（不做任何修改） """
        state_input = Input(shape=(self.state_dim,), name="State_Input")
        cont_action_input = Input(shape=(self.action_dim_continuous,), name="Cont_Action_Input")

        # 状态归一化
        normalized_state = Lambda(
            lambda x: (x - self.state_low) / self.state_range,
            name="State_Normalization"
        )(state_input)

        if self.action_dim_discrete > 0:
            disc_action_input = Input(shape=(self.action_dim_discrete,), name="Disc_Action_Input")
            concat = Concatenate(axis=-1, name="Concat_Input")([normalized_state, cont_action_input, disc_action_input])
            inputs = [state_input, cont_action_input, disc_action_input]
        else:
            concat = Concatenate(axis=-1, name="Concat_Input")([normalized_state, cont_action_input])
            inputs = [state_input, cont_action_input]

        x = Dense(units[0], kernel_initializer=tf.keras.initializers.he_uniform(), name="L0")(concat)
        x = LeakyReLU()(x)
        x = LayerNormalization(name="LayerNorm0")(x)
        for index in range(1, len(units)):
            x = Dense(units[index], kernel_initializer=tf.keras.initializers.he_uniform(), name=f"L{index}")(x)
            x = LeakyReLU()(x)
            x = Dropout(0.1, name=f"Dropout{index}")(x)

        qc_value_output = Dense(1, activation="linear", name="Q_Constraint_Output")(x)
        return Model(inputs=inputs, outputs=qc_value_output)

    # 软更新
    def update_target_weights(self, model, target_model, tau=np.float32(0.01)):
        weights = model.get_weights()
        target_weights = target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
        target_model.set_weights(target_weights)

if __name__ == "__main__":
    print("Imports worked!")
