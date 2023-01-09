import torch
import torch.nn.functional as F
from cProfile import label
import matplotlib
matplotlib.use('Agg')
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import copy

def fit(data,  model, optimizer, loss_fn, losses_dict, final_epochs, bs):

    epoch_train_loss, epoch_val_loss, total_train_loss_list, total_val_loss_list = losses_dict['epoch_train_loss'], losses_dict['epoch_val_loss'], losses_dict['total_train_loss_list'], losses_dict['total_val_loss_list']
    train_percentage = 0.8
    train_dataset, val_dataset = torch.utils.data.random_split(data, [int(len(data)*train_percentage), len(data)-int(len(data)*train_percentage)])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = bs, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = bs, shuffle = not True)
    log_interval = 25
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    global_weight_collector = copy.deepcopy(list(model.parameters()))

    for epoch in range(final_epochs):

        print('============ EPOCH {}/{} ============'.format(epoch+1, final_epochs))
        epoch_start_time = time.time()
        
        print('TRAINING')

        model.train()
        class_cnt = data.get_class_cnt()
        running_train_loss = 0
        train_loss_list = []

        sigmoid = torch.nn.Sigmoid()
        total = 0
        correct = 0

        start_time = time.time()
        for batch_idx, (img, target) in enumerate(train_loader):
            # print(type(img), img.shape) # , np.unique(img))

            img = img.to(device)
            target = target.float().to(device)
            
            optimizer.zero_grad()    
            
            if data.get_name() == 'NIH' or data.get_name() == 'ChexPert':
                out = model(img)  
                loss = loss_fn(out, target)
                running_train_loss += loss*img.shape[0]
                train_loss_list.append(loss.item())
                preds = np.round(sigmoid(out).cpu().detach().numpy())
                targets = target.cpu().detach().numpy()
                total += len(targets)*class_cnt
                correct += (preds == targets).sum()
            elif data.get_name() == 'CIFAR10' or data.get_name() == 'CIFAR100':
                out = torch.log_softmax(model(img), dim=1)
                target = torch.argmax(target, dim=1)
                loss = loss_fn(out, target)
                running_train_loss += loss*img.shape[0]
                train_loss_list.append(loss.item())
                prediction = out.max(1, keepdim=True)[1] # index
                preds = prediction.squeeze().cpu().detach().numpy()
                targets = target.cpu().detach().numpy()
                total += len(targets)
                correct += (preds == targets).sum()
            
            ############
            # for fedprox
            fed_prox_reg = 0.0
            for param_index, param in enumerate(model.parameters()):
                fed_prox_reg += ((0.45 / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss = loss + fed_prox_reg
            ########

            loss.backward()
            optimizer.step()

            if (batch_idx+1)%log_interval == 0:
                batch_time = time.time() - start_time
                m, s = divmod(batch_time, 60)
                print('Train Loss for batch {}/{} @epoch{}/{}: {} in {} mins {} secs'.format(str(batch_idx+1).zfill(3), str(len(train_loader)).zfill(3), epochs_till_now, final_epoch, round(loss.item(), 5), int(m), round(s, 2)))
            
            start_time = time.time()
        
        mean_running_train_loss = running_train_loss/float(len(train_loader.dataset))
        train_loss = train_loss_list

        print('VALIDATION')

        model.eval()
        running_val_loss = 0
        val_loss_list = []
        val_loader_examples_num = len(val_loader.dataset)
        sigmoid = torch.nn.Sigmoid()

        total = 0
        correct = 0
        total_target = []
        total_preds = []

        if data.get_name() == 'NIH' or data.get_name() == 'ChexPert':
            probs = np.zeros((val_loader_examples_num, class_cnt), dtype = np.float32)
            gt    = np.zeros((val_loader_examples_num, class_cnt), dtype = np.float32)
            k=0

        with torch.no_grad():
            batch_start_time = time.time()    
            for batch_idx, (img, target) in enumerate(val_loader):

                img = img.to(device)
                target = target.to(device)    
        
                if data.get_name() == 'NIH' or data.get_name() == 'ChexPert':
                    out = model(img)       
                    loss = loss_fn(out, target)    
                    preds = np.round(sigmoid(out).cpu().detach().numpy())
                    targets = target.cpu().detach().numpy()
                    total += len(targets)*class_cnt
                    correct += (preds == targets).sum()
                    probs[k: k + out.shape[0], :] = out.cpu()
                    gt[   k: k + out.shape[0], :] = target.cpu()
                    k += out.shape[0]
                elif data.get_name() == 'CIFAR10' or data.get_name() == 'CIFAR100':
                    out = torch.log_softmax(model(img), dim=1)        
                    target = torch.argmax(target, dim=1)
                    loss = loss_fn(out, target)    
                    prediction = out.max(1, keepdim=True)[1]
                    preds = prediction.squeeze().cpu().detach().numpy()
                    targets = target.cpu().detach().numpy()
                    total += len(targets)
                    correct += (preds == targets).sum()

                running_val_loss += loss.item()*img.shape[0]
                val_loss_list.append(loss.cpu().detach().numpy())

                if ((batch_idx+1)%log_interval == 0):
                    batch_time = time.time() - batch_start_time
                    m, s = divmod(batch_time, 60)
                    print('Val Loss   for batch {}/{} @epoch{}/{}: {} in {} mins {} secs'.format(str(batch_idx+1).zfill(3), str(len(val_loader)).zfill(3), epoch, final_epochs, round(loss.item(), 5), int(m), round(s, 2)))
                
                batch_start_time = time.time()  
                total_target+= targets.tolist()
                total_preds += sigmoid(out).cpu().detach().numpy().tolist()
                
        # metric scenes
        mean_running_val_loss = running_val_loss/float(len(val_loader.dataset))
        val_loss = val_loss_list
        
        print("Test Accuracy: ", correct/total)
        try:
            roc_auc = roc_auc_score(gt, probs)
        except:
            roc_auc = 0

        epoch_train_loss.append(mean_running_train_loss)
        epoch_val_loss.append(mean_running_val_loss)

        total_train_loss_list.extend(train_loss)
        total_val_loss_list.extend(val_loss)

        print('\nTRAIN LOSS : {}'.format(mean_running_train_loss))
        print('VAL   LOSS : {}'.format(mean_running_val_loss))
        print('VAL ROC_AUC: {}'.format(roc_auc))

        total_epoch_time = time.time() - epoch_start_time
        m, s = divmod(total_epoch_time, 60)
        h, m = divmod(m, 60)
        print('\nEpoch {}/{} took {} h {} m'.format(epoch, final_epochs, int(h), int(m)))

    return model.state_dict()