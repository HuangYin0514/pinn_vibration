import os

from utils import count_parameters, load_network


from .pinn import PINN


__model_factory = {
    "PINN": PINN,
}


def choose_model(net_name, *args, **kwargs):
    if net_name not in __model_factory.keys():
        raise ValueError("Model '{}' is not implemented".format(net_name))
    net = __model_factory[net_name](*args, **kwargs)
    return net


def get_model(config, logger, *args, **kwargs):
    net = choose_model(config.net_name, config, logger)

    logger.info("=================>")
    logger.info("Start get models...")
    logger.info("{} loaded".format(config.net_name))
    logger.info("Number of parameters in model: {}".format(count_parameters(net)))

    if os.path.exists(config.load_net_path):
        load_network(net, config.load_net_path, config.device)
        logger.info("Network loaded from '{}'".format(config.load_net_path))
    else:
        logger.warning("Network not found in '{}'".format(config.load_net_path))
        logger.warning("Weight of network not be loaded ")

    return net.to(config.device).to(config.dtype)
