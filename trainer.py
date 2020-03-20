import torch

import channels
from channels import awgn, energy_constraint


def train(trainloader, net, optimizer, loss_func, device, loss_vec, batch_size, EbN0_dB_train, args):
    running_loss = 0.0
    running_corrects = 0

    net.train()
    for step, (x, y) in enumerate(trainloader):  # gives batch data

        # Move batches to GPU
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()  # clear gradients for this training step

        # This helps us export the messages at each stage and view how they evolve on Tensorboard.
        # Alternatively, we can just say output = net(x) if we just want to compute the final output
        x_transmitted = net.transmitter(x)
        x_normalized = channels.energy_constraint(x_transmitted, args)
        x_noisy = channels.awgn(x_normalized, args, EbN0_dB_train, device)
        output = net.receiver(x_noisy)
        loss = loss_func(output, y)  # Apply cross entropy loss

        # Backward and optimize
        loss.backward()  # back propagation, compute gradients
        optimizer.step()  # apply gradients
        loss_vec.append(loss.item())  # Append to loss_vec
        # exp_lr_scheduler.step()

        pred_labels = torch.max(output, 1)[1].data.squeeze()
        accuracy = sum(pred_labels == y) / float(batch_size)

        # statistics
        running_loss += loss.item()
        running_corrects += accuracy
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model

    train_epoch_loss = running_loss / step
    train_epoch_acc = running_corrects/ step
    return train_epoch_loss, train_epoch_acc

def validate(net,valloader,loss_func, batch_size, device, EbN0_dB_train,args):
    net.eval()
    with torch.no_grad():
        for val_data, val_labels in valloader:
            val_data = val_data.to(device)
            val_labels = val_labels.to(device)

            val_encoded_signal = net.transmitter(val_data)
            val_constrained_encoded_signal = channels.energy_constraint(val_encoded_signal, args)
            val_noisy_signal = channels.awgn(val_constrained_encoded_signal, args, EbN0_dB_train, device)
            val_decoded_signal = net.receiver(val_noisy_signal)

            val_loss = loss_func(val_decoded_signal, val_labels)  # Apply cross entropy loss
            val_pred_labels = torch.max(val_decoded_signal, 1)[1].data.squeeze()
            val_accuracy = sum(val_pred_labels == val_labels) / float(batch_size)
    return val_loss,  val_accuracy


def test(net, testloader, device, EbN0_test, args):
    net.eval()
    with torch.no_grad():
        for test_data, test_labels in testloader:
            test_data = test_data.to(device)
            test_labels = test_labels.to(device)

            encoded_signal = net.transmitter(test_data)
            constrained_encoded_signal = channels.energy_constraint(encoded_signal, args)
            noisy_signal = channels.awgn(constrained_encoded_signal, args, EbN0_test, device)
            decoded_signal = net.receiver(noisy_signal)

            pred_labels = torch.max(decoded_signal, 1)[1].data.squeeze()
            test_BLER = sum(pred_labels != test_labels) / float(test_labels.size(0))
    return test_BLER

