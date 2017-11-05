import tensorflow as tf


def forward(self, samples):

    # we get batch_size x num1 matrix
    with tf.name_scope("fc1_hidden"):
        hidden_value1 = tf.nn.relu(tf.matmul(samples, self.fc1_weights) + self.fc1_biases)
    with tf.name_scope("fc2_hidden"):
        hidden_value2 = tf.nn.relu(tf.matmul(hidden_value1, self.fc2_weights) + self.fc2_biases)

    # we get batch_size x 10 matrix(logits)
    with tf.name_scope("logits"):
        logits = tf.matmul(hidden_value2, self.fc3_weights) + self.fc3_biases
        return logits