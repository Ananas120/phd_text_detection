# Copyright (C) 2023 Langlois Quentin, ICTEAM, UCLouvain. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tensorflow as tf

from custom_architectures.current_blocks import Conv2DBN

def Bottleneck(x, filters, use_bias = False, strides = 1, expansion = 4, dilation = 1, name = 'bottleneck'):
    inputs = x
    if isinstance(x, tuple):
        inputs = tf.keras.layers.Input(shape = x, name = 'input')
    
    residual = inputs
    out = inputs
    for i, (f, k) in enumerate([(filters, 1), (filters, 3), (filters * expansion, 1)]):
        if k > 2:
            out = tf.keras.layers.ZeroPadding2D(((k // 2, k // 2), (k // 2, k // 2)))(out)
        out = tf.keras.layers.Conv2D(
            f, kernel_size = k, strides = strides if k != 1 else 1, use_bias = use_bias, name = '{}/conv{}'.format(name, i + 1)
        )(out)
        out = tf.keras.layers.BatchNormalization(epsilon = 1e-5, momentum = 0.1, name = '{}/bn{}'.format(name, i + 1))(out)
        if i < 2:
            out = tf.keras.layers.ReLU(name = '{}/relu/{}'.format(name, i + 1))(out)
    
    if strides != 1 or filters * expansion != residual.shape[-1]:
        residual = tf.keras.layers.Conv2D(
            filters * expansion, kernel_size = 1, use_bias = use_bias, strides = strides, name = '{}/downsample/0'.format(name)
        )(residual)
        residual = tf.keras.layers.BatchNormalization(
            epsilon = 1e-5, momentum = 0.1, name = '{}/downsample/1'.format(name)
        )(residual)
    
    out = tf.keras.layers.Add(name = '{}/add'.format(name))([out, residual])
    out = tf.keras.layers.ReLU(name = '{}/relu3'.format(name))(out)
    
    return out if not isinstance(x, tuple) else tf.keras.Model(inputs, out)

def ResNet152(x, n_blocks = [3, 8, 36], filters = [64, 128, 256], strides = [1, 2, 1], name = 'resnet'):
    inputs = x
    if isinstance(x, tuple):
        inputs = tf.keras.layers.Input(shape = x, name = 'input_image')
    
    out = tf.keras.layers.ZeroPadding2D(((3, 3), (3, 3)))(inputs)
    out = tf.keras.layers.Conv2D(64, kernel_size = 7, strides = 2, use_bias = False, name = '{}/conv1'.format(name))(out)
    out = tf.keras.layers.BatchNormalization(epsilon = 1e-5, momentum = 0.1, name = '{}/bn1'.format(name))(out)
    out = tf.keras.layers.ReLU()(out)
    
    out = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))(out)
    out = tf.keras.layers.MaxPooling2D(pool_size = 3, strides = 2)(out)
    
    for i, (n, f, s) in enumerate(zip(n_blocks, filters, strides)):
        for j in range(n):
            out = Bottleneck(out, filters = f, strides = s if j == 0 else 1, name = '{}/{}/{}'.format(name, i + 4, j))
    
    return out if not isinstance(x, tuple) else tf.keras.Model(inputs, out)

def SSD(input_shape = (512, 512, 3),
        backbone    = 'ResNet152',
        filters     = [256, 256, 128, 128, 128, 128],
        kernel_size = [3, 3, 3, 3, 3, 4],
        
        nb_box      = [4, 6, 6, 6, 4, 4, 4],
        
        name        = 'SSD'
       ):
    inputs = tf.keras.layers.Input(shape = input_shape, name = 'input_image')
    
    if backbone.lower() == 'resnet152':
        feature_extractor = ResNet152(x = input_shape, name = 'feature_extractor')
    
    features = feature_extractor(inputs)
    
    out     = features
    outputs = [features]
    for i, (f, k) in enumerate(zip(filters, kernel_size)):
        out = tf.keras.layers.Conv2D(f, kernel_size = 1, use_bias = False, name = '{}/additional_blocks/{}/0'.format(name, i))(out)
        out = tf.keras.layers.BatchNormalization(epsilon = 1e-5, momentum = 0.1, name = '{}/additional_blocks/{}/1'.format(name, i))(out)
        out = tf.keras.layers.ReLU()(out)

        out = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))(out)
        out = tf.keras.layers.Conv2D(
            f * 2, kernel_size = k, use_bias = False, strides = 2, name = '{}/additional_blocks/{}/3'.format(name, i)
        )(out)
        out = tf.keras.layers.BatchNormalization(
            epsilon = 1e-5, momentum = 0.1, name = '{}/additional_blocks/{}/4'.format(name, i)
        )(out)
        out = tf.keras.layers.ReLU()(out)
        outputs.append(out)
    
    locs = []
    conf = []
    for i, (feat, f) in enumerate(zip(outputs, nb_box)):
        loc_out = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))(feat)
        loc_out = tf.keras.layers.Conv2D(f * 4, kernel_size = 3, name = '{}/loc/{}'.format(name, i))(loc_out)
        loc_out = tf.keras.layers.Lambda(
            lambda x: tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [tf.shape(x)[0], 4, -1])
        )(loc_out)
        
        conf_out = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))(feat)
        conf_out = tf.keras.layers.Conv2D(f * 2, kernel_size = 3, name = '{}/confs/{}'.format(name, i))(conf_out)
        conf_out = tf.keras.layers.Lambda(
            lambda x: tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [tf.shape(x)[0], 2, -1])
        )(conf_out)

        locs.append(loc_out)
        conf.append(conf_out)
    
    outputs = [
        tf.keras.layers.Concatenate(axis = -1)(locs),
        tf.keras.layers.Concatenate(axis = -1)(conf)
    ]
    
    return tf.keras.Model(inputs, outputs, name = name)
    
def TextBoxes():
    pass

custom_functions    = {
    'Bottleneck' : Bottleneck,
    'TextBoxes'  : TextBoxes,
    'SSD' : SSD
}
