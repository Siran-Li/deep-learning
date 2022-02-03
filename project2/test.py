import torch
import math

from model import MLP
from optim import SGD
from loss import MSELoss


def generate_disc_set(nb):
    input = torch.empty(nb, 2).uniform_(0, 1)
    msign = -(input-torch.tensor([0.5, 0.5])).pow(2).sum(1).sub(1/(2*math.pi)).sign()
    target = msign.add(1).div(2).long()

    return input, target

def train(model, train_input, train_target, batch_size, nb_epochs, lr):
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=lr)

    losses = []
    for e in range(nb_epochs):
        running_loss = 0.0
        for b in range(0, train_input.size(0), batch_size):
            out = model(train_input[b:b+batch_size])

            target = train_target[b:b+batch_size].view(-1, 1).float()
            loss = criterion(out, target)

            optimizer.zero_grad()
            dldy = criterion.backward()
            model.backward(dldy)
            optimizer.step()
            
            running_loss += loss.item()

        epoch_loss = running_loss / (b / batch_size + 1)
        losses.append(epoch_loss)
        if (e+1) % 10 == 0:
            print("Epoch {}, loss: {} ".format(e, epoch_loss))
    
def test(model, test_input, test_target, batch_size):
    nb_acc = 0
    
    for b in range(0, test_input.size(0), batch_size):
        output = model(test_input[b:b+batch_size])

        output = output.view(-1)
        pred = output.round()

        target = test_target[b:b+batch_size]
        nb_acc = nb_acc + (target==pred).sum()
    
    return nb_acc/test_input.size(0)


def main():
    train_input, train_target = generate_disc_set(1000)
    test_input, test_target = generate_disc_set(1000)

    mean, std = train_input.mean(), train_input.std()

    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    batch_size = 100
    nb_epochs = 300
    lr = 0.1

    model = MLP()

    train(model, train_input, train_target, batch_size, nb_epochs, lr)
    acc = test(model, test_input, test_target, batch_size)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main()