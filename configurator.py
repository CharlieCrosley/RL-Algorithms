import sys
from config.templates import VPGConfig
import importlib

def get_config():
    config = VPGConfig() # default config since vpg is the default algorithm
    for arg in sys.argv[1:]:
        if arg.startswith('config/'):
            s = arg.replace("/", ".")[:-3]
            config = importlib.import_module(s).config
            break

    for arg in sys.argv[1:]:
        # Override config args from command line
        if arg.startswith('--'):
            key, val = arg.split('=')
            key = key[2:]

            try:
                val = eval(val) # type cast
            except NameError:
                pass # value is a string
        
            setattr(config, key, val)

    # Print the config
    for k, v in config.__dict__.items():
        print(f"{k} = {v}")

    return config