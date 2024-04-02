#!python
from multiprocessing import Process, Manager
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def convert_to_normal_dict(mp_dict):
    normal_dict = {}
    manager = Manager()
    manager_dict_type = type(manager.dict())
    manager_list_type = type(manager.list())
    for key, value in mp_dict.items():
        if isinstance(value, manager_dict_type):
            normal_dict[key] = convert_to_normal_dict(value)
        elif isinstance(value, manager_list_type):
            normal_dict[key] = list(value)
            for i, item in enumerate(normal_dict[key]):
                if isinstance(item, manager_dict_type):
                    normal_dict[key][i] = convert_to_normal_dict(item)
        else:
            normal_dict[key] = value
    return normal_dict