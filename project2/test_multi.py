import torch
import math
from loss import MSELoss
from optim import SGD
from model import MLP
import numpy as np


def generate_disc_set(nb):
    input = torch.empty(nb, 2).uniform_(0, 1)
    msign = -(input-torch.tensor([0.5, 0.5])).pow(2).sum(1).sub(1/(2*math.pi)).sign()
    target = msign.add(1).div(2).long()

    return input, target

def train(model, train_input, train_target, batch_size, nb_epochs, lr):
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=lr)
    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), batch_size):
            output = model(train_input[b:b+batch_size])

            target = train_target[b:b+batch_size].view(-1, 1).float()
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            dldy = criterion.backward()
            model.backward(dldy)
            optimizer.step()
            
        if (e+1) % 10 == 0:
            print("Epoch {}, batch {}, loss: {} ".format(e, b/batch_size, loss.item()))


def test(model, test_input, test_target, batch_size):
    nb_acc = 0
    
    for b in range(0, test_input.size(0), batch_size):
    
        output = model(test_input[b:b+batch_size])

        output = output.view(-1)
        pred = output.round()

        target = test_target[b:b+batch_size]
        nb_acc = nb_acc + (target==pred).sum()
    
    return nb_acc/test_input.size(0)


def fit(train_input, train_target, test_input, test_target, batch_size, nb_epochs, lr):
    rounds = 10
    accs = []

    for i in range(rounds):
        model = MLP()
        print("Evaluating for round {}".format(i))

        # TODO Do not use this function, no data shuffle
        p = torch.randperm(len(train_input))
        train_input = train_input[p]
        train_target = train_target[p]

        train(model, train_input, train_target, batch_size, nb_epochs, lr)
        acc = test(model, test_input, test_target, batch_size)
        print("Round {} accuracy: {}".format(i, acc))
        accs.append(acc.item())
    
    accs = np.array(accs)
    print("Accuracy mean: {}, std: {}".format(accs.mean(), accs.std()))
    


def main():
    train_input, train_target = generate_disc_set(1000)
    test_input, test_target = generate_disc_set(1000)

    mean, std = train_input.mean(), train_input.std()

    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    batch_size = 100
    nb_epochs = 200
    lr = 0.1
    
    fit(train_input, train_target, test_input, test_target, \
        batch_size, nb_epochs, lr)


if __name__ == "__main__":
    main()