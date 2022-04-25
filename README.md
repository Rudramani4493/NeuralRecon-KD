# NeuralRecon-KD


## Changes made for NeuralRecon-Student

1. Within neuralrecon.py have renamed the class NeuralRecon to NeuralReconStudent
2. Within init.py renamed NeuralRecon to NeuralReconStudent as we have changed the class name there.
3. Within neucon_network.py have reduced the number of channels and also the number of layers within conv0 of Mnasnet to reduce the number of parameters.



import torch
import torch.nn as nn
from distillation.hintonDistiller import HintonDistiller
from distillation.utils import MLP, PseudoDataset

# Initialize random models and distiller
student = NeuralReconStudent('cfg')
teacher = NeuralReconTeacher('cfg')
distiller = HintonDistiller(alpha=0.5, studentLayer=-2, teacherLayer=-2)

# Initialize objectives and optimizer
objective = nn.CrossEntropyLoss()
distillObjective = nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.SGD(student.parameters(), lr=0.1)

# Pseudo dataset and dataloader 
trainloader = torch.utils.data.DataLoader(''' Yet to be added for scannet ''')

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
