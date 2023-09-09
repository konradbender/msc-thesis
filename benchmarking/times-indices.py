from python.glauber.DataStructs.ListDict import ListDict
import numpy as np
import random
import timeit


class MyNaiveList:
    
    def __init__(self) -> None:
        self.set = set()
        
    def add(self, item):
        self.set.add(item)
    
    def remove(self, item):
        self.set.remove(item)  
        
    def choose_random_item(self):
        return random.choice(list(self.set))

def run_test(list, id1, id2, id3):
    for index in id1:
        index = tuple(index)
        list.add(index)
        
    # try adding each one again
    for index in id2:
        index = tuple(index)
        list.add(index)
    
    # sample random items from the list
    for i in range(int(1e4)):
        list.choose_random_item()
        
    for index in id3:
        index = tuple(index)
        list.remove(index)
        
if __name__ == '__main__':
    
    SHAPE = 100
    
    x = np.arange(0, SHAPE)
    y = np.arange(0, SHAPE)
    
    # create indices and three copies that are shuffled to make sure
    # that the order of the indices does not matter
    id1  = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    id2 = id1.copy()
    id3 = id2.copy()
    np.random.shuffle(id1)
    np.random.shuffle(id2)
    np.random.shuffle(id3)
    
    ld = ListDict(None, None)
    
    print("Averge time for ListDict:")
    x = timeit.timeit(lambda: run_test(ld, id1, id2, id3), number=10)
    print(x)
    print("Averge time for Naive:")
    x = timeit.timeit(lambda: run_test(MyNaiveList(), id1, id2, id3), number=10)
    print(x)
    
