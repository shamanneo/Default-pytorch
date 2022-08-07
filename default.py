from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.onnx

class Net(nn.Module) :

    pass 

class MainApp :

    #   Member variables.

    __args = None 
    __device = None 

    __train_kwargs = None 
    __test_kwargs = None 
    __train_loader = None 
    __test_loader = None 

    __optimizer = None 
    __scheduler = None 

    #   Member functions.

    def __init__(self) :
        self.SetTrainingArgs()

    def SetTrainingArgs(self) :
        parser = argparse.ArgumentParser(description = "Handwritten digit classification")
        parser.add_argument("--batch-size", type = int, default = 64, metavar = "N",
                             help = "input batch size for training (default : 64)" )
        parser.add_argument("--test-batch-size", type = int, default = 1000, metavar = "N",
                        help = "input batch size for testing (default : 1000)" )
        parser.add_argument("--epochs", type = int, default = 10, metavar = "N",
                            help = "number of epochs to train (default : 14)")
        parser.add_argument("--lr", type = float, default = 1.0, metavar = "LR",
                            help = "learning rate (default: 1.0)")
        parser.add_argument("--gamma", type = float, default = 0.7, metavar = "M",
                            help = "Learning rate step gamma (default: 0.7)")
        parser.add_argument("--no-cuda", action = "store_true", default = False,
                            help = "disables CUDA training")
        parser.add_argument("--dry-run", action = "store_true", default = False,
                            help = "quickly check a single pass")
        parser.add_argument("--seed", type = int, default = 1, metavar = "S",
                            help = "random seed (default: 1)")
        parser.add_argument("--log-interval", type = int, default = 10, metavar = "N",
                            help = "how many batches to wait before logging training status")
        parser.add_argument("--save-model", action = "store_true", default = False,
                            help = "For Saving the current Model")
    
        # Additional arguments here...

        self.__args = parser.parse_args()

    def PrepareTraining(self) :
        use_cuda = not self.__args.no_cuda and torch.cuda.is_available()
        torch.manual_seed(self.__args.seed)
        self.__device = torch.device("cuda" if use_cuda else "cpu")

        self.__train_kwargs = { "batch_size" : self.__args.batch_size } 
        self.__test_kwargs = { "batch_size" : self.__args.test_batch_size } 
        if(use_cuda) :
            cuda_kwargs = { "num_workers" : 1, "pin_memory" : True, "shuffle" : True }
            self.__train_kwargs.update(cuda_kwargs)
            self.__test_kwargs.update(cuda_kwargs)
    
    def LoadDataset(self) :
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        pass 
        # Load dataset...

    def Train(self, epoch) :
        self.__model.train()
        for batch_idx, (data, target) in enumerate(self.__train_loader) :
            data = data.to(self.__device)
            target = target.to(self.__device)
            self.__optimizer.zero_grad()
            output = self.__model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.__optimizer.step()
            if(batch_idx % self.__args.log_interval == 0) :
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.__train_loader.dataset),
                    100. * batch_idx / len(self.__train_loader), loss.item()))
                if(self.__args.dry_run) :
                    break

    def Test(self) :
        self.__model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad() :
            for data, target in self.__test_loader:
                data, target = data.to(self.__device), target.to(self.__device)
                output = self.__model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim = 1, keepdim = True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.__test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.__test_loader.dataset),
            100. * correct / len(self.__test_loader.dataset)))
    
    def Save(self, path) :
        torch.save(self.__model.state_dict(), path)
        print("The model has been saved successfully.")

    def Run(self) :
        # pre-procesing
        self.PrepareTraining()
        self.LoadDataset()
        # main-processing
        self.__model = Net().to(self.__device)
        self.__optimizer = optim.Adadelta(self.__model.parameters(), lr = self.__args.lr)
        self.__scheduler = StepLR(self.__optimizer, step_size = 1, gamma = self.__args.gamma)

        for epoch in range(1, self.__args.epochs + 1) :
            self.Train(epoch)
            self.Test()
            self.__scheduler.step()

        # model saving.
        if(self.__args.save_model == True) :
            path = None
            self.Save(path)


def main() :
    main_app = MainApp() 
    main_app.Run()

if(__name__ == "__main__") :
    main() 
