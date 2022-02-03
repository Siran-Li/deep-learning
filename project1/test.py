from dlc_practical_prologue import generate_pair_sets

import torch
import torch.nn as nn
from torch import optim

from models import MLP, ConvNet


def train(model, train_input, train_target, train_classes, batch_size, nb_epochs, lr, lambd=1, auxiliary=False):
    model.train()
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if auxiliary:
        # We are using auxiliary loss, 3 losses will be computed.
        for e in range(nb_epochs):
            for b in range(0, train_input.size(0), batch_size):
                out_1, out_2, out_f = model(train_input[b:b+batch_size])

                target = train_target[b:b+batch_size].view(-1, 1).float()
                loss_f = criterion2(out_f, target)

                classes = train_classes[b:b+batch_size]
                loss_1 = criterion1(out_1, classes[:, 0])
                loss_2 = criterion1(out_2, classes[:, 1])

                loss = loss_1 + loss_2 + lambd*loss_f

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if (e+1) % 10 == 0:
                print("Epoch {}, batch {}, loss: {} ".format(e, b/batch_size, loss.item()))
        
    else:
        # We are not using auxiliary loss, 1 loss will be computed.
        for e in range(nb_epochs):
            for b in range(0, train_input.size(0), batch_size):
                output = model(train_input[b:b+batch_size])

                target = train_target[b:b+batch_size].view(-1, 1).float()
                loss = criterion2(output, target)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if (e+1) % 10 == 0:
                print("Epoch {}, batch {}, loss: {} ".format(e, b/batch_size, loss.item()))


def test(model, test_input, test_target, test_classes, batch_size, auxiliary=False):
    model.eval()
    nb_acc = 0
    
    for b in range(0, test_input.size(0), batch_size):
        if auxiliary:
            _, _, output = model(test_input[b:b+batch_size])
        else:
            output = model(test_input[b:b+batch_size])

        output = output.view(-1)
        pred = output.round()

        target = test_target[b:b+batch_size]
        nb_acc = nb_acc + (target==pred).sum()
    
    return nb_acc/test_input.size(0)


def fit(train_input, train_target, train_classes, test_input, test_target, test_classes, batch_size, nb_epochs, lr, weight_share, auxiliary, use_mlp):
    if use_mlp:
        print("Evaluating MLP...")
    else:
        print("Evaluating ConvNet...")
    
    #TODO rounds should be at least 10
    rounds = 10
    accs = []

    for i in range(rounds):
        print("Evaluating for round {}".format(i))
        if use_mlp:
            #TODO Modify the config to the best architecture
            config = {'out_dims':[128, 64, 32, 256]}
            model = MLP(config, weight_share, auxiliary, activate='relu', p=0)
        else:
            #TODO Modify the config to the best architecture
            config = {'chns':[32, 64, 32], 'n_hid': 512}
            model = ConvNet(config, weight_share, auxiliary, activate='relu')
        
        #Shuffle data before training
        p = torch.randperm(len(train_input))
        train_input = train_input[p]
        train_target = train_target[p]
        train_classes = train_classes[p]

        if use_mlp:
            # Set the best tradeoff lambda between losses
            lambd = 0.9
        else:
            lambd = 0.8

        train(model, train_input, train_target, train_classes, batch_size, nb_epochs, lr, lambd=lambd, auxiliary=auxiliary)
        acc = test(model, test_input, test_target, test_classes, batch_size, auxiliary=auxiliary)
        print("Round {} accuracy: {}".format(i, acc))
        accs.append(acc.item())
    
    accs = torch.tensor(accs)
    print("Accuracy mean: {}, std: {}".format(accs.mean(), accs.std()))
    

def main():
    # generate training and test data
    nb_pairs = 1000
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(nb_pairs)
    
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)

    # define hyperparameters
    batch_size = 100
    nb_epochs = 30
    lr = 1e-3
    #lr = 1e-2       # for sigmoid

    weight_share = True
    auxiliary = True
    use_mlp = False
    
    fit(train_input, train_target, train_classes, test_input, test_target, test_classes, \
        batch_size, nb_epochs, lr, weight_share, auxiliary, use_mlp)


if __name__ == '__main__':
    main()