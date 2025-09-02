import numpy as np

class Rewarder:
    def __init__(self, r_search, r_wait) -> None:
        self._r_search, self._r_wait = -np.inf(), -np.inf()
        self.r_search = r_search
        self.r_wait = r_wait

    @property
    def r_search(self):
        return self._r_search
    
    @property
    def r_wait(self):
        return self._r_wait
    
    @r_wait.setter
    def r_wait(self, r):
        if self.r_search < r:
            raise ValueError('r_search must be greater than r_wait.')
        self._r_search = r
    
    @r_search.setter
    def r_search(self, r):
        if r < self.r_wait:
            raise ValueError('r_search must be greater than r_wait.')
        self._r_search = r