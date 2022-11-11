import os


class Predict:
    def __init__(self, data, model, device):
        self.data = data
        self.model = model
        self.device = device
        self.model.to(device)
        
    def predict(self):
        print('Start recognizing the images...')
        self.model.eval()
        print('Loaded model. ')
        
        predict_opt = []
        
        for img, img_name in self.data:
            
            img = img.to(self.device)
            
            opt = self.model(img)
            
            opt = opt.argmax(dim=-1).item()
            
            predict_opt.append(opt)
            
            os.rename('./dataset/unlabeled_imgs/'+img_name[0],
                      './dataset/labeled_imgs/'+str(opt)+'_'+img_name[0])

        print('Predict done. ')
        print(predict_opt)
        print('Recognization displayed in the file name. ')
        