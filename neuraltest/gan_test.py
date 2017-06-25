from copy import deepcopy
from unittest import mock
import tensorflow as tf

from gan import discriminator
from gan import generator
from gan import model_inputs
from gan_util import model_loss
from gan_util import model_opt

def test_safe(func):
    def func_wrapper(*args):
        with tf.Graph().as_default():
            result = func(*args)
        print("Test passed")
        return result
    return func_wrapper
    
def _assert_tensor_shape(tensor, shape, display_name):
    assert tf.assert_rank(tensor, len(shape), message="{} has wrong rank.".format(
            display_name))
    tensor_shape = tensor.get_shape().as_list() if len(shape) else []
    wrong_dimension = [ten_dim for ten_dim, cor_dim in zip(tensor_shape, shape)
                        if cor_dim is not None and ten_dim != cor_dim]
    assert not wrong_dimension, "{} has wrong shape. Found {}".format(display_name,
                                 tensor_shape)

def _check_input(tensor, shape, display_name, tf_name=None):
    assert tensor.op.type == "Placeholder", "{} is not a Placeholder.".format(
            display_name)
    _assert_tensor_shape(tensor, shape, "Real Input")
    if tf_name:
        assert tensor.name == tf_name, "{} has bad name. Found name {}".format(
                display_name, tensor.name)
     
class TempoMock():
    def __init__(self, module, attribute_name):
        self.original_attribute = deepcopy(getattr(module, attribute_name))
        setattr(module, attribute_name, mock.MagicMock())
        self.module = module
        self.attribute_name = attribute_name
    def __enter__(self):
        return getattr(self.module, self.attribute_name)
    def __exit__(self, type, value, traceback):
        setattr(self.module, self.attribute_name, self.original_attribute)

@test_safe
def test_model_inputs(model_inputs):
    vector_size = 3 # (x, y, z)
    input_real, input_z, input_label, learn_rate = model_inputs(vector_size)
    _check_input(input_real, [None, vector_size], "Real Input")
    _check_input(input_z, [None, vector_size], "Z Input")
    _check_input(input_label, [None, 1], "Label Input")
    _check_input(learn_rate, [], "Learning Rate")

@test_safe   
def test_discriminator(discriminator, tf_module):
    with TempoMock(tf_module, "variable_scope") as mock_variable_scope:
        vector = tf.placeholder(tf.float32, [None, 3])
        labels = tf.placeholder(tf.float32, [None, 1])
        output, logits = discriminator(vector, labels, [[4, 128],[128, 64],[64, 1]]) #
        _assert_tensor_shape(output, [None, 1], 
                             "Discriminator Training (reuse=false) output")
        _assert_tensor_shape(logits, [None, 1], 
                             "Discriminator Training (reuse=false) logits")
        assert mock_variable_scope.called, \
            "tf.variable_scope not called in Discriminator Training (reuse=false)"
        
        mock_variable_scope.reset_mock()
        
        output_reuse, logits_reuse = discriminator(vector, labels, [[4, 128],[128, 64],[64, 1]], reuse=True) #
        _assert_tensor_shape(output_reuse, [None, 1], 
                             "Discriminator Inference (reuse=true) output")
        _assert_tensor_shape(logits_reuse, [None, 1], 
                             "Discriminator Inference (reuse=true) logits")
        assert mock_variable_scope.called, \
            "tf.variable_scope not called in Discriminator Inference (reuse=true)"
        assert mock_variable_scope.call_args == mock.call("discriminator", reuse=True), \
            "tf.variable_scope called with wrong arguments in Discriminator Inference (reuse=True)"

@test_safe
def test_generator(generator, tf_module):
    with TempoMock(tf_module, "variable_scope") as mock_variable_scope:
        z = tf.placeholder(tf.float32, [None, 3])
        labels = tf.placeholder(tf.float32, [None, 1])
        output = generator(z, labels, [[4, 128],[128, 64],[64, 3]])
        _assert_tensor_shape(output, [None, 3], "Generator output (is_train=True)")
        assert mock_variable_scope.called, \
            "tf.variable_scope not called in Generator Training (reuse=false)"
        assert mock_variable_scope.call_args == mock.call("generator", reuse=False), \
            "tf.variable_scope called with wrong arguments in Generator Training (reuse=false)"
        
        mock_variable_scope.reset_mock()
        
        output = generator(z, labels, [[4, 128],[128, 64],[64, 3]], is_train=False)
        _assert_tensor_shape(output, [None, 3], 
                             "Generator output (is_train=False)")
        assert mock_variable_scope.called, \
            "tf.variable_scope not called in Generator Training (reuse=True)"
        assert mock_variable_scope.call_args == mock.call("generator", reuse=True), \
            "tf.variable_scope called with wrong arguments in Generator Training (reuse=True)"

@test_safe
def test_model_loss(model_loss):
    hidden_layer_shape_generator = [[4, 128],[128, 64],[64, 3]]
    hidden_layer_shape_discriminator = [[4, 128],[128, 64],[64, 1]]
    input_real = tf.placeholder(tf.float32, [None, 3])
    input_z = tf.placeholder(tf.float32, [None, 3])
    input_label = tf.placeholder(tf.float32, [None, 1])
    d_loss, g_loss = model_loss(input_real, input_z, input_label, hidden_layer_shape_generator, hidden_layer_shape_discriminator)
    _assert_tensor_shape(d_loss, [], "Discriminator Loss")
    _assert_tensor_shape(g_loss, [], "Generator Loss")

@test_safe
def test_model_opt(model_opt, tf_module):
    with TempoMock(tf_module, "trainable_variables") as mock_trainable_variables:
        with tf.variable_scope("discriminator"):
            discriminator_logits = tf.Variable(tf.zeros([3, 3]))
        with tf.variable_scope("generator"):
            generator_logits = tf.Variable(tf.zeros([3, 3]))
        mock_trainable_variables.return_value = [discriminator_logits, generator_logits]
        d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=discriminator_logits,
                labels=[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=generator_logits,
                labels=[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
        d_train_opt, g_train_opt = model_opt(d_loss, g_loss)
        assert mock_trainable_variables.called, "tf.mock_trainable_variables not called"
    
if __name__ == "__main__":
    test_model_inputs(model_inputs)
    test_discriminator(discriminator, tf)
    test_generator(generator, tf)
    test_model_loss(model_loss)
    test_model_opt(model_opt, tf)