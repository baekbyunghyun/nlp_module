import tensorflow as tf
import matplotlib.pyplot as plt


class CustomScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, wramup_steps=4000):
        super(CustomScheduler, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = wramup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


if __name__ == '__main__':
    sample_learning_rate = CustomScheduler(d_model=128)

    plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
    plt.xlabel("Train Step")
    plt.ylabel("Learning Rate")
    plt.show()
