import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, LeakyReLU,BatchNormalization, Dropout, LayerNormalization


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
        # self.action_dim_discrete = action_dim_discrete
        self.action_dim_discrete = 0
        self.action_bound = action_bound
        self.action_shift = action_shift

        # 状态的上下限
        self.state_low = np.array(state_low,dtype=np.float32)
        self.state_high = np.array(state_high,dtype=np.float32)
        self.state_range = self.state_high - self.state_low


    def actor(self, units=(512, 256, 64, 32), tau_=0.5):
        state_input = Input(shape=(self.state_dim,), name="State_Input")

        # 归一化状态
        normalized_state = Lambda(
            lambda x: (x - self.state_low) / self.state_range,
            name="State_Normalization"
        )(state_input)

        # 隐藏层
        x = Dense(units[0], kernel_initializer=tf.keras.initializers.he_uniform(),name="L0")(normalized_state)
        x = LeakyReLU()(x)
        for index in range(1, len(units)):
            x = Dense(units[index], kernel_initializer=tf.keras.initializers.he_uniform(),name=f"L{index}")(x)
            x = LeakyReLU()(x)

        ## **连续动作：随机策略**
        # 计算均值和标准差
        # mu_output = Dense(self.action_dim_continuous,name="Cont_Output_Mean")(x)
        # sigma_output = Dense(self.action_dim_continuous, activation="softplus", name="Cont_Output_StdDev")(x)  # 这个限制范围>0

        mu_output = Dense(self.action_dim_continuous,
                          kernel_initializer=tf.keras.initializers.he_uniform(),
                          name="Cont_Output_Mean")(x)
        mu_output = Lambda(lambda x: tf.tanh(x), name="Cont_Output_Mean_Tanh")(mu_output)  # ✅ 限制 mu 在 [-1,1] ,这里必须这么做，你要保证和动作一致

        sigma_output = Dense(self.action_dim_continuous, activation="softplus",
                             kernel_initializer=tf.keras.initializers.he_uniform(),
                             name="Cont_Output_StdDev")(x)

        # ✅ sigma不要接近于0
        sigma_output = Lambda(lambda x: tf.clip_by_value(x, 0.1, np.inf))(sigma_output)

        # 重新参数化技巧（Reparameterization Trick）给出连续动作
        # 写成可微形式，方便求导
        epsilon = Lambda(lambda x: tf.random.normal(shape=tf.shape(x)))(sigma_output)
        sampled_cont_action = Lambda(lambda x: x[0] + x[1] * x[2])([mu_output, sigma_output, epsilon])
        sampled_cont_action = Lambda(lambda x: tf.tanh(x))(sampled_cont_action)  # ✅ 最终动作归一化，经过tanh

        ## **离散动作（如果 action_dim_discrete > 0）**
        if self.action_dim_discrete > 0:
            # 预测 logits（未归一化概率）
            logits = Dense(self.action_dim_discrete,
                           kernel_initializer=tf.keras.initializers.he_uniform(),
                           name="Disc_Output_Logits")(x)

            # Gumbel-Softmax 采样：离散动作范围属于0-1
            def gumbel_softmax_sample(logits, tau_=0.5):
                """ Gumbel-Softmax 采样函数 """
                uniform_noise = tf.random.uniform(tf.shape(logits), minval=1e-8, maxval=1)
                gumbel_noise = -tf.math.log(-tf.math.log(uniform_noise))
                return tf.nn.softmax((logits + gumbel_noise) / tau_)

            sampled_disc_action = Lambda(lambda x: gumbel_softmax_sample(x, tau_), name="Disc_Output_Gumbel")(logits)

            model = Model(inputs=state_input,
                          outputs=[mu_output, sigma_output, sampled_cont_action, sampled_disc_action,logits])
        else:
            model = Model(inputs=state_input, outputs=[mu_output, sigma_output, sampled_cont_action])

        return model

        ## **🚀 双 Q Critic 网络**

    # 输入原始动作和actor得出的动作
    def critic(self, units=(128, 128, 32)):
        """ **创建两个独立的 Q 网络：Q1 和 Q2** """

        def build_q_network():
            state_input = Input(shape=(self.state_dim,), name="State_Input")
            cont_action_input = Input(shape=(self.action_dim_continuous,), name="Cont_Action_Input")

            # **状态归一化**
            normalized_state = Lambda(
                lambda x: (x - self.state_low) / self.state_range,
                name="State_Normalization"
            )(state_input)

            if self.action_dim_discrete > 0:
                disc_action_input = Input(shape=(self.action_dim_discrete,), name="Disc_Action_Input")
                concat = Concatenate(axis=-1, name="Concat_Input")(
                    [normalized_state, cont_action_input, disc_action_input])
                inputs = [state_input, cont_action_input, disc_action_input]
            else:
                concat = Concatenate(axis=-1, name="Concat_Input")([normalized_state, cont_action_input])
                inputs = [state_input, cont_action_input]

            # **隐藏层**
            x = Dense(units[0], kernel_initializer=tf.keras.initializers.he_uniform(), name="L0")(concat)
            x = LeakyReLU()(x)
            x = LayerNormalization(name="LayerNorm0")(x)  # 🚀 增加 Layer Normalization
            for index in range(1, len(units)):
                x = Dense(units[index], kernel_initializer=tf.keras.initializers.he_uniform(), name=f"L{index}")(x)
                x = LeakyReLU()(x)
                x = Dropout(0.1, name=f"Dropout{index}")(x)  # 🚀 添加 Dropout

            # **单独的 Q 值输出**
            q_value_output = Dense(1, activation="linear", name="Q_Value_Output")(x)


            return Model(inputs=inputs, outputs=q_value_output)

        # **创建两个 Q 网络**
        q1_model = build_q_network()
        q2_model = build_q_network()

        return q1_model, q2_model

        ## **🚀 约束 Critic Q_C**

    def constraint_critic(self, units=(128, 128, 32)):
        """ **创建单独的约束 Critic Q_C** """
        state_input = Input(shape=(self.state_dim,), name="State_Input")
        cont_action_input = Input(shape=(self.action_dim_continuous,), name="Cont_Action_Input")

        # **状态归一化**
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

        # ✅ 逐层构建 Q_c 网络（加 LayerNorm & Dropout 防止过拟合）
        x = Dense(units[0], kernel_initializer=tf.keras.initializers.he_uniform(), name="L0")(concat)
        x = LeakyReLU()(x)
        x = LayerNormalization(name="LayerNorm0")(x)  # 🚀 增加 Layer Normalization
        for index in range(1, len(units)):
            x = Dense(units[index], kernel_initializer=tf.keras.initializers.he_uniform(), name=f"L{index}")(x)
            x = LeakyReLU()(x)
            x = Dropout(0.1, name=f"Dropout{index}")(x)  # 🚀 添加 Dropout

        # **约束 Q 值输出**
        qc_value_output = Dense(1, activation="linear", name="Q_Constraint_Output")(x)

        return Model(inputs=inputs, outputs=qc_value_output)

    def update_target_weights(self, model, target_model, tau=np.float32(0.01)):
        weights = model.get_weights()
        target_weights = target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
        target_model.set_weights(target_weights)



if __name__=="__main__":
    print("Imports worked!")