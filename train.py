import torch.cuda
from torch import nn


class Train:
    def __init__(self, data, model, num_epoch, learning_rate, device):
        self.data = data
        self.model = model
        self.num_epoch = num_epoch
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.device = device
        self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self):
        train_accs = []
        train_losses = []
        print('Start training...')

        for epoch in range(self.num_epoch):
            self.model.train()
            for i, data in enumerate(self.data):
                img, label = data
                img, label = img.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()
                
                opt = self.model(img)
                
                loss = batch_loss = self.criterion(opt, label)
                
                acc = (opt.argmax(dim=-1) == label).float().mean()
                train_losses.append(loss.item())
                train_accs.append(acc)
                
                batch_loss.backward()
                self.optimizer.step()
            
            train_loss = sum(train_losses) / len(train_losses)
            train_acc = sum(train_accs) / len(train_accs)
            print(
                'Epoch: (' + str(epoch + 1) + '/' + str(self.num_epoch) + '), Loss: {:.5f}'.format(train_loss),
                'Accuracy: {:.5f}'.format(train_acc.item())
            )
        
        print('Training finished. ')
        
        model_path = './model.pth'
        torch.save(self.model, model_path)
        print('Model saved in "' + model_path + '". ')
