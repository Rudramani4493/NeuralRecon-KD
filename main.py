import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from loguru import logger
from distillation.hintonDistiller import HintonDistiller
from distillation.utils import MLP, PseudoDataset
from NeuralRecon-Student.models import NeuralReconTeacher
from NeuralRecon-Teacher.models import NeuralReconStudent
from distillation import cfg, update_cfg
from datasets import transforms, find_dataset_def

# Initialize random models and distiller
student = NeuralReconStudent(cfg)
teacher = NeuralReconTeacher(cfg)
distiller = HintonDistiller(alpha=0.5, studentLayer=-2, teacherLayer=-2)

# Initialize objectives and optimizer
objective = nn.CrossEntropyLoss()
distillObjective = nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
 
# dataset, dataloader
MVSDataset = find_dataset_def(cfg.DATASET)
train_dataset = MVSDataset(cfg.TRAIN.PATH, "train", transforms, cfg.TRAIN.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1)
test_dataset = MVSDataset(cfg.TEST.PATH, "test", transforms, cfg.TEST.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1)


if cfg.DISTRIBUTED:
    train_sampler = DistributedSampler(train_dataset, shuffle=False)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=cfg.TRAIN.N_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        sampler=test_sampler,
        num_workers=cfg.TEST.N_WORKERS,
        pin_memory=True,
        drop_last=False
    )
else:
    trainloader = DataLoader(train_dataset, cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.TRAIN.N_WORKERS,
                                drop_last=True)
    testloader = DataLoader(test_dataset, cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.TEST.N_WORKERS,
                               drop_last=False)

'''trainloader = torch.utils.data.DataLoader(
    PseudoDataset(size=(100)),
    batch_size=512,
    shuffle=True)'''

# Load state if checkpoint is provided
checkpoint = None
startEpoch = distiller.load_state(checkpoint, student, teacher, optimizer)
epochs = 15

# Construct tensorboard logger
distiller.init_tensorboard_logger()

for epoch in range(startEpoch, epochs+1):
        # Training step for one full epoch
        trainMetrics = distiller.train_step(student=student,
                                            teacher=teacher,
                                            dataloader=trainloader,
                                            optimizer=optimizer,
                                            objective=objective,
                                            distillObjective=distillObjective)
        
        # Validation step for one full epoch
        validMetrics = distiller.validate(student=student,
                                          dataloader=trainloader,
                                          objective=objective)
        metrics = {**trainMetrics, **validMetrics}
        
        # Log to tensorbard
        distiller.log(epoch, metrics)

        # Save model
        distiller.save(epoch, student, teacher, optimizer)
        
        # Print epoch performance
        distiller.print_epoch(epoch, epochs, metrics)
