''' Hydra utilities that require the usage of cycles and if statements.

(Hydra .yaml files do not allow its usage.)
'''
from hydra.core.hydra_config import HydraConfig


def get_only_swept_params():
    '''From all the overriden variable, gets only the ones inside the params_sweep yaml.
    
    It does not include the seed.

    Output example:
    'lr=0.001,weight_decay=0.2'
    '''

    swept_params = HydraConfig.get().sweeper.params.keys()
    override_params = HydraConfig.get().overrides.task

    swept_params_list = []
    for override_param in override_params:
        key = override_param.split('=')[0]
        if key in swept_params and key != 'seed':
            swept_params_list.append(override_param)

    return ','.join(swept_params_list)
