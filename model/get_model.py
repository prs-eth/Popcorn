from model.popcorn import POPCORN

model_dict = {
    "POPCORN": POPCORN
}

def get_model_kwargs(args, model_name):
    """
    :param args: arguments
    :param model_name: name of the model
    :return: kwargs for the model
    """

    # kwargs for the model
    kwargs = {
        'input_channels': args.Sentinel1 * 2 + args.NIR * 1 + args.Sentinel2 * 3,
        'feature_extractor': args.feature_extractor
    }

    kwargs['occupancymodel'] = args.occupancymodel
    kwargs['pretrained'] = args.pretrained 
    kwargs['biasinit'] = args.biasinit
    kwargs['sentinelbuildings'] = args.sentinelbuildings
    return kwargs
