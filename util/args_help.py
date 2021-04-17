import os
from argparse import ArgumentParser
from enum import Enum, EnumMeta
import logging

logger = logging.getLogger(__name__)


def fill_from_dict(defaults, a_dict):
    for arg, val in a_dict.items():
        d = defaults.__dict__[arg]
        if type(d) is tuple:
            d = d[0]
        if isinstance(d, Enum):
            defaults.__dict__[arg] = type(d)[val]
        elif isinstance(d, EnumMeta):
            defaults.__dict__[arg] = d[val]
        else:
            defaults.__dict__[arg] = val


def fill_from_args(defaults):
    """
    Builds an argument parser, parses the arguments, updates and returns the object 'defaults'
    :param defaults: an object with fields to be filled from command line arguments
    :return:
    """
    parser = ArgumentParser()
    # if defaults has a __required_args__ we set those to be required on the command line
    required_args = []
    if hasattr(defaults, '__required_args__'):
        required_args = defaults.__required_args__
        for reqarg in required_args:
            if reqarg not in defaults.__dict__:
                raise ValueError(f'argument "{reqarg}" is required, but not present in __init__')
            if reqarg.startswith('_'):
                raise ValueError(f'arguments should not start with an underscore ({reqarg})')
    for attr, value in defaults.__dict__.items():
        # ignore members that start with '_'
        if attr.startswith('_'):
            continue

        # if it is a tuple, we assume the second is the help string
        help_str = None
        if type(value) is tuple and len(value) == 2 and type(value[1]) is str:
            help_str = value[1]
            value = value[0]

        # check if it is a type we can take on the command line
        if type(value) not in [str, int, float, bool] and not isinstance(value, Enum) and not isinstance(value, type):
            raise ValueError(f'Error on {attr}: cannot have {type(value)} as argument')
        if type(value) is bool and value:
            raise ValueError(f'Error on {attr}: boolean arguments (flags) must be false by default')

        # also handle str to enum conversion
        t = type(value)
        if isinstance(value, Enum):
            t = str
            value = value.name
        elif isinstance(value, EnumMeta):
            t = type
            value = str

        if t is type:
            # indicate a required arg by specifying a type rather than value
            parser.add_argument('--'+attr, type=value, required=True, help=help_str)
        elif t is bool:
            # support bool with store_true (required false by default)
            parser.add_argument('--'+attr, default=False, action='store_true', help=help_str)
        else:
            parser.add_argument('--'+attr, type=t, default=value, help=help_str, required=(attr in required_args))
    args = parser.parse_args()
    # now update the passed object with the arguments
    fill_from_dict(defaults, args.__dict__)
    # call _post_argparse() if the method is defined
    try:
        defaults._post_argparse()
    except AttributeError:
        pass
    return defaults
