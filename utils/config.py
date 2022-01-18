import collections
import importlib.util
import os

import six

__all__ = ['Config', 'configs', 'update_configs_from_module', 'update_configs_from_arguments']

class G(dict):
    def __getattr__(self, k):
        if k not in self:
            raise AttributeError(k)
        return self[k]
    def __setattr__(self, k, v):
        self[k] = v
    def __delattr__(self, k):
        del self[k]

class Config(G):
    def __init__(self, func=None, **kwargs):
        super().__init__(**kwargs)

        if func is not None and not callable(func):
            raise TypeError('func "{}" is not a callable function or class'.format(repr(func)))
        
        self.__dict__['_func_'] = func

    def __call__(self, *args, **kwargs):
        if self._func_ is None:
            return self

        # override kwargs
        # 因为class G, config的attribute就是config的key
        # config的key和value就成了__func__的参数
        for k, v in self.items():
            kwargs.setdefault(k, v)

        return self._func_(*args, **kwargs)

    def __str__(self, indent=0):
        text = ''
        if self._func_ is not None:
            text += ' ' * indent + '[func] = ' + str(self._func_)
            extra = False
            
            text += ')\n' if extra else '\n'
        for k, v in self.items():
            text += ' ' * indent + '[' + str(k) + ']'
            if isinstance(v, Config):
                text += '\n' + v.__str__(indent + 2)
            else:
                text += ' = ' + str(v)
            text += '\n'

        while text and text[-1] == '\n':
            text = text[:-1]
        return text

    def __repr__(self):
        text = ''
        if self._func_ is not None:
            text += repr(self._func_)

        return text

    @staticmethod
    def update_from_modules(*modules):
        for module in modules:
            module = module.replace('.py', '').replace('/', '.')
            importlib.import_module(module)

    @staticmethod
    def update_from_arguments(*args):
        update_configs_from_arguments(args)


configs = Config()

def update_configs_from_arguments(args):
    index = 0

    while index < len(args):
        arg = args[index]

        if arg.startswith('--configs.'):
            arg = arg.replace('--configs.', '')
        else:
            raise Exception('unrecognized argument "{}"'.format(arg))

        if '=' in arg:
            index, keys, val = index + 1, arg[:arg.index('=')].split('.'), arg[arg.index('=') + 1:]
        else:
            index, keys, val = index + 2, arg.split('.'), args[index + 1]

        config = configs
        # 处理连环'.'
        for k in keys[:-1]:
            if k not in config:
                config[k] = Config()
            config = config[k]

        def parse(x):
            if (x[0] == '\'' and x[-1] == '\'') or (x[0] == '\"' and x[-1] == '\"'):
                return x[1:-1]
            try:
                x = eval(x)
            except:
                pass
            return x

        config[keys[-1]] = parse(val)