class Parameter(object):
    def __init__(self, data, requires_grad=True, skip_decay=False):
        self.data = data
        self.grad = 0
        self.grad_p = 0
        # self.skip_decay = skip_decay
        # self.requires_grad = requires_grad
