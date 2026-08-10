# @Time   : 2022/3/2
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
recbole_pjf.config
##########################
"""

import os
from recbole.config import Config
from recbole_pjf.utils import get_model

class PJFConfig(Config):

    def __init__(self, model=None, dataset=None, config_file_list=None, config_dict=None):
        super(PJFConfig, self).__init__(model, dataset, config_file_list, config_dict)

    def _load_internal_config_dict(self, model, model_class, dataset):
        super(PJFConfig, self)._load_internal_config_dict(model, model_class, dataset)
        current_path = os.path.dirname(os.path.realpath(__file__))
        overall_init_file = os.path.join(current_path, './properties/overall.yaml')
        model_init_file = os.path.join(current_path, './properties/model/' + model + '.yaml')
        dataset_init_file = os.path.join(current_path, './properties/dataset/' + dataset + '.yaml')

        for file in [overall_init_file, model_init_file, dataset_init_file]:
            if os.path.isfile(file):
                config_dict = self._update_internal_config_dict(file)
                if file == dataset_init_file:
                    self.parameters['Dataset'] += [
                        key for key in config_dict.keys() if key not in self.parameters['Dataset']
                    ]

        self.internal_config_dict['MODEL_TYPE'] = model_class.type

    def _get_model_and_dataset(self, model, dataset):
        if model is None:
            try:
                model = self.external_config_dict['model']
            except KeyError:
                raise KeyError(
                    'model need to be specified in at least one of the these ways: '
                    '[model variable, config file, config dict, command line] '
                )
        if not isinstance(model, str):
            final_model_class = model
            final_model = model.__name__
        else:
            final_model = model
            final_model_class = get_model(final_model)

        if dataset is None:
            try:
                final_dataset = self.external_config_dict['dataset']
            except KeyError:
                raise KeyError(
                    'dataset need to be specified in at least one of the these ways: '
                    '[dataset variable, config file, config dict, command line] '
                )
        else:
            final_dataset = dataset

        return final_model, final_model_class, final_dataset

    def _set_default_parameters(self):
        super(PJFConfig, self)._set_default_parameters()
        default_max_sent_num = 10
        default_max_sent_len = 10
        default_max_longsent_len = 64
        if 'max_sent_num' not in self.final_config_dict:
            self.final_config_dict['max_sent_num'] = default_max_sent_num
        if 'max_sent_len' not in self.final_config_dict:
            self.final_config_dict['max_sent_len'] = default_max_sent_len
        if 'max_longsent_len' not in self.final_config_dict:
            self.final_config_dict['max_longsent_len'] = default_max_longsent_len
        if 'prediction_head' not in self.final_config_dict:
            self.final_config_dict['prediction_head'] = 'elem_mlp'

        # BGNN parameters —— 注册到 RecBole 使其能从 YAML 中加载
        bgnn_defaults = {
            'bgnn_weight_decay': 0.1,
            'bgnn_dropout': 0.3,
            'bgnn_batch_num_u': 8,
            'bgnn_batch_num_v': 8,
            'bgnn_layer_depth': 2,
            'bgnn_epochs': 1,
            'bgnn_learning_type': 'inference',
            'bgnn_dis_hidden_dim': 1,
            'bgnn_batch_size': 100,
            'bgnn_v_attr_dimensions': 64,
            'bgnn_learning_rate': 0.001,
            'bgnn_u_num': 256,
            'bgnn_u_attr_dimensions': 64,
            'bgnn_v_num': 256,
        }
        for key, default_val in bgnn_defaults.items():
            if key not in self.final_config_dict:
                self.final_config_dict[key] = default_val

    def change_direction(self):
        self.final_config_dict['USER_ID_FIELD'], self.final_config_dict['ITEM_ID_FIELD'] = \
            self.final_config_dict['ITEM_ID_FIELD'], self.final_config_dict['USER_ID_FIELD']
