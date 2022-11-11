class Test:
    def __init__(self, data, model, device):
        self.data = data
        self.model = model
        self.device = device
        self.model.to(device)
        
    def test(self):
        print('Start testing...')
        self.model.eval()
        print('Loaded model. ')
        test_accs = []
        
        for img, label in self.data:
            img, label = img.to(self.device), label.to(self.device)
            opt = self.model(img)
            
            test_acc = (opt.argmax(dim=-1) == label).float().mean()
            test_accs.append(test_acc)
        
        test_acc = (sum(test_accs) / len(test_accs)).float()
        print('Test done. ')
        
        print('test_set accuracy is: '+str(test_acc.item()))
        