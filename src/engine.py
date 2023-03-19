from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm


class Trainer:
    
    best_val_loss: float
    
    def __init__(self, 
            model: torch.nn.Module, 
            optimizer: torch.optim.Optimizer, 
            criterion: torch.nn.modules.loss._Loss,
            train_dataset: Dataset, 
            val_dataset: Optional[Dataset] = None, 
            scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
            batch_size: int = 128,
            ):
        """Класс для обучение PyTorch моделей

        Parameters
        ----------
        model : torch.nn.Module
            Модель
        optimizer : torch.optim.Optimizer
            Оптимизатор
        criterion : torch.nn.modules.loss._Loss
            Класс ошибки
        train_dataset : Dataset
            Датасет для обучения
        val_dataset : Optional[Dataset], optional
            Датасет для валидации, при наличии будет считаться лосс на нем, и сохраняться модель с лучшим лоссом, by default None
        scheduler : Optional[torch.optim.lr_scheduler.LRScheduler], optional
            Планировщик learning rate, by default None
        batch_size : int, optional
            Размер батча, by default 128
        """
        self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.batch_size = batch_size

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)

        self.global_step = 0        
        self.writer = SummaryWriter()
        
    def train(self, num_epochs) -> None:
        train_loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=True, batch_size=self.batch_size)

        # валидационный датасет не стоит перемешивать
        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(self.val_dataset, shuffle=False, pin_memory=True, batch_size=self.batch_size)
        
        self.best_val_loss = float('inf')
        
        for epoch in tqdm(range(num_epochs), desc='Training:'):
            
            self.model.train()
            for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
                self.training_step(batch)
            
            if val_loader is not None:
                self.validate(val_loader)
                
    def training_step(self, batch) -> None:
        self.optimizer.zero_grad()
        loss = self.compute_loss_on_batch(batch)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
            
        # Запись логов
        self.writer.add_scalar(
            "loss", 
            ..., 
            global_step=self.global_step,
        )
        
        self.writer.add_scalar(
            "lr", 
            ..., 
            global_step=self.global_step,
        )
        
        self.global_step += 1
        
    def validate(self, val_loader: DataLoader) -> None:
        self.model.eval()
        
        val_losses = []
        for batch in tqdm(val_loader, desc='Validating'):
            loss = self.compute_loss_on_batch(batch)
            val_losses.append(loss.item())
        
        # Сохранение лучшей модели
        # (можете реализовать если хотите)
        val_loss = np.mean(val_losses)        
        if val_loss < self.best_val_loss:
            self.save_checkpoint()
            self.best_val_loss = val_loss
            
        self.writer.add_scalar(
            "val_loss", 
            ..., 
            global_step=self.global_step,
        )

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        ...

    def compute_loss_on_batch(self, batch) -> torch.Tensor:
        # Считает лосс и метрики
        ...
    

def evaluate_loader(loader: DataLoader, model: torch.nn.Module) -> Dict[str, float]:
    
    with torch.no_grad():
        model.eval()
        
        # <YOUR CODE>
        
        return {
            'acc': ...,
            'f1': ...,
        }
