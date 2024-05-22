
import tensorflow as tf
import numpy as np

class RotaryEmbedding:
    def __init__(self, seq_len, dim, max_wavelength=10000, scaling_factor=1.0, sequence_axis=1, feature_axis=-1):
        self.seq_len = seq_len
        self.dim = dim
        self.max_wavelength = max_wavelength
        self.scaling_factor = scaling_factor
        self.sequence_axis = sequence_axis
        self.feature_axis = feature_axis

    def _get_inverse_freq(self, rotary_dim):
        freq_range = np.arange(0, rotary_dim, 2, dtype=np.float) / (rotary_dim*1.0)
        inverse_freq = 1.0 / (self.max_wavelength**freq_range)
        return inverse_freq

    def _compute_positions(self, inputs, start_index=0.0):
        positions = tf.range(start=0, limit=self.seq_len)
        return positions + start_index

    def _compute_cos_sin_embedding(self, inputs, start_index=0, positions=None):
        rotary_dim = inputs.get_shape().as_list()[-1]
        inverse_freq = self._get_inverse_freq(rotary_dim)
        if positions is None:
            positions = self._compute_positions(inputs, start_index)
        positions = tf.cast(positions, dtype=tf.float32) / self.scaling_factor
        inverse_freq = tf.cast(inverse_freq, tf.float32)
        inverse_freq = tf.convert_to_tensor(inverse_freq)
        freq = tf.einsum("i,j->ij", positions, inverse_freq)
        embedding = tf.stack([freq, freq], axis=-2)
        embedding = tf.reshape(embedding, (self.seq_len, self.dim))

        embedding = tf.expand_dims(embedding, axis=0)
        cos_emb = tf.cast(tf.cos(embedding), tf.float32)
        sin_emb = tf.cast(tf.sin(embedding), tf.float32)

        return cos_emb, sin_emb

    def _apply_rotary_pos_emb(self, tensor, cos_emb, sin_emb):
        x1, x2 = tf.split(tensor, 2, axis=-1)
        half_rot_tensor = tf.stack([-x2, x1], axis=-2)
        half_rot_tensor = tf.reshape(half_rot_tensor, tf.shape(tensor))
        return (tensor * cos_emb) + (half_rot_tensor * sin_emb)


    def __call__(self, inputs, start_index=0, positions=None):
        # inputs: [B, N, D]
        cos_emb, sin_emb = self._compute_cos_sin_embedding(inputs, start_index, positions)
        print(f'cos_emb: {cos_emb}, sin_emb: {sin_emb}')
        outputs = self._apply_rotary_pos_emb(inputs, cos_emb, sin_emb)
        return outputs


batch_size = 1
dim = 4
seq_len = 3
num_heads = 1

tensor = tf.ones([batch_size, seq_len, dim], dtype=tf.float32)
rot_emb_layer = RotaryEmbedding(seq_len=tf.shape(tensor)[1], dim=tf.shape(tensor)[-1], scaling_factor=2.0)
tensor_rot = rot_emb_layer(tensor)

print(tensor_rot)

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(tensor_rot))
