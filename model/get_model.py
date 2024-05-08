"""
Project: 🍿POPCORN: High-resolution Population Maps Derived from Sentinel-1 and Sentinel-2 🌍🛰️
Nando Metzger, 2024
"""

from model.popcorn import POPCORN
from typing import Dict, Any, NamedTuple

class Args(NamedTuple):
    Sentinel1: bool
    NIR: bool
    Sentinel2: bool
    feature_extractor: str
    occupancymodel: bool
    pretrained: bool
    biasinit: float
    sentinelbuildings: bool

model_dict = {
    "POPCORN": POPCORN
}

def calculate_input_channels(args: Args) -> int:
    """ Calculate the number of input channels based on the presence of Sentinel and NIR data. """
    channels = 0
    if args.Sentinel1:
        channels += 2
    if args.NIR:
        channels += 1
    if args.Sentinel2:
        channels += 3
    return channels

def get_model_kwargs(args: Args, model_name: str) -> Dict[str, Any]:
    """
    Construct keyword arguments for a model based on input args and model name.
    
    Args:
        args (Args): The configuration arguments for the model.
        model_name (str): The name of the model to get kwargs for.

    Returns:
        Dict[str, Any]: A dictionary containing keyword arguments for the model.

    Raises:
        ValueError: If the model_name is not in the model dictionary.
    """
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} not found in model dictionary")

    # kwargs for the model
    kwargs = {
        'input_channels': calculate_input_channels(args),
        'feature_extractor': args.feature_extractor,
        'occupancymodel': args.occupancymodel,
        'pretrained': args.pretrained,
        'biasinit': args.biasinit,
        'sentinelbuildings': args.sentinelbuildings
    }

    return kwargs