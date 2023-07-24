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
import numpy as np
import pandas as pd
import tensorflow as tf

from models.detection.med_unet import MedUNet

class MedUNetClusterer(MedUNet):
    def __init__(self, labels = None, embedding_dim = 32, normalize = True, distance_metric = 'euclidian', ** kwargs):
        self.distance_metric = distance_metric
        self.embedding_dim   = embedding_dim
        self.normalize       = normalize
        
        kwargs.update({'labels' : None, 'nb_class' : None})
        super().__init__(** kwargs)
    
    def _build_model(self, * args, final_activation = None, ** kwargs):
        super()._build_model(
            * args,
            #transfer_kwargs  = {'partial_transfer' : False},
            final_activation = final_activation,
            normalize_output = self.normalize,
            ** kwargs
        )
    
    @property
    def last_dim(self):
        return self.embedding_dim
    
    @property
    def last_output_dim(self):
        return None
    
    def __str__(self):
        des = super().__str__()
        des += "- Embedding dim   : {}\n".format(self.embedding_dim)
        des += "- Distance metric : {}\n".format(self.distance_metric)
        return des
    
    def compile(self, loss = 'GE2ESegLoss', loss_config = {}, ** kwargs):
        loss_config['distance_metric'] = self.distance_metric
        super().compile(loss = loss, loss_config = loss_config, ** kwargs)
    
    def filter_data(self, inputs, output):
        if tf.shape(output.indices)[0] == 0:
            tf.print('Empty slice :', tf.shape(inputs), tf.shape(output), tf.shape(output.indices))
        return tf.shape(output.indices)[0] > 0
    
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            'normalize'       : self.normalize,
            'embedding_dim'   : self.embedding_dim,
            'distance_metric' : self.distance_metric
        })
        return config
    
