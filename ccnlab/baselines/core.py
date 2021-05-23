from abc import ABC
import numpy as np

class Model(ABC):

    def reset(self):
        ''' Erase all memory from the model '''

    def act(self, cs, ctx, us, t):
        ''' Perform an action given current stimulus input 
        :param cs: conditioned stimulus
        :param ctx: context
        :param us: unconditioned stimulus
        :param t: timestep within trial
        :return: (un)conditioned response
        '''
