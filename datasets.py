import torch
import torch.utils.data as Data

def prepare_data(set_size, class_num, batch_size):
    '''
    trainloader, valloader and testloader are iterators used for batching, shuffling
    and loading the data in parallel using multiprocessing workers
    '''
    labels = (torch.rand(set_size) * class_num).long()
    data = torch.sparse.torch.eye(class_num).index_select(dim=0, index=labels)
    dataset = Data.TensorDataset(data, labels)
    loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataset, loader, labels
