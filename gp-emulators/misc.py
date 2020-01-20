
class gmresCounter(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.numIter = 0
    def __call__(self, rk=None):
        self.numIter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

