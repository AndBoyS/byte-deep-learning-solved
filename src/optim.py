import warnings

from torch.optim.lr_scheduler import _LRScheduler
import warnings


class WarmupScheduler(_LRScheduler):
    def __init__(self,
                 optimizer,
                 warmup_steps=0,
                 initial_lr_factor=1e-7,
                 last_epoch=-1,
                 verbose=False):
        self.warmup_steps = warmup_steps
        self.initial_lr_factor = initial_lr_factor
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                'To get the last learning rate computed by the scheduler, '
                'please use `get_last_lr()`.', UserWarning)

        if self.last_epoch == 0:
            return [
                group['lr'] *
                (self.initial_lr_factor if self.warmup_steps > 0 else 1)
                for group in self.optimizer.param_groups
            ]
        else:
            if self.last_epoch > self.warmup_steps:
                return [group['lr'] for group in self.optimizer.param_groups]
            else:
                eta = self.initial_lr_factor
                t = self.warmup_steps
                return [
                    group['lr'] * (t * eta + (1 - eta) * self.last_epoch) /
                    (t * eta + (1 - eta) * (self.last_epoch - 1))
                    for group in self.optimizer.param_groups
                ]
