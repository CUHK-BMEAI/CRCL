import copy
import logging
import random

import numpy as np
import os
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from inc_net_v2 import SimpleVitNet
from utils.toolkit import target2onehot, tensor2numpy, accuracy

num_workers = 8

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments=[]
        self._network = None
        self._device = args["device"][0]

    def eval_task(self):
        y_pred, y_true, pred = self._eval_conservative(self.test_loader)
        acc_total,grouped = self._evaluate(y_pred, y_true)
        grouped_list = []
        grouped_list.append(grouped) 

        if self.args["merge_result"] and self._cur_task > 0: 
            grouped_list = []
            _, _, pred_c = self._eval_conservative(self.test_loader)       
            _, pred_r = self._eval_rapid(self.test_loader)   

            pred_c = torch.from_numpy(pred_c)
            pred_r = torch.from_numpy(pred_r)
            T = 0.1  
            probs_c = F.softmax(pred_c / T, dim=1)
            probs_r = F.softmax(pred_r / T, dim=1)

            kl_1 = F.kl_div(probs_c.log(), probs_r, reduction='none').sum(dim=1)
            kl_2 = F.kl_div(probs_r.log(), probs_c, reduction='none').sum(dim=1)
            kl_div = 0.5 * (kl_1 + kl_2)

            threshold = torch.quantile(kl_div, 0.5)
            combine = kl_div > threshold

            eps = 1e-8
            entropy_c = -(probs_c * torch.log(probs_c + eps)).sum(dim=1)
            entropy_r = -(probs_r * torch.log(probs_r + eps)).sum(dim=1)
            inv_entropy_c = 1.0 / (entropy_c + eps)
            inv_entropy_r = 1.0 / (entropy_r + eps)
            weight_c = inv_entropy_c / (inv_entropy_c + inv_entropy_r)
            weight_r = 1.0 - weight_c

            combined_logits = weight_c.unsqueeze(1) * pred_c + weight_r.unsqueeze(1) * pred_r
            final_pred = torch.where(
                combine.unsqueeze(1), combined_logits,
                torch.where(
                    probs_c.max(dim=1).values.unsqueeze(1) > probs_r.max(dim=1).values.unsqueeze(1),
                    pred_c,
                    pred_r
                )
            )

            y_pred_tmp = torch.topk(final_pred, k=1, dim=1)[1].numpy()
            acc_total_tmp, grouped_tmp = self._evaluate(y_pred_tmp, y_true)
            
            acc_total, grouped, y_pred = acc_total_tmp, grouped_tmp, y_pred_tmp

            grouped_list.append(grouped)

        print(acc_total, grouped_list)
        return acc_total, grouped_list, y_pred, y_true
    
    def _eval_rapid(self, loader):
        self.model_branch1.eval()
        y_pred = []
        pred = []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self.model_branch1(inputs)["logits"]
            predicts = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1] 
            y_pred.append(predicts.cpu().numpy())
            pred.append(outputs.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(pred)  

    def _eval_conservative(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        pred = []
        features = []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
                feature = self._network.convnet(inputs)
            predicts = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1] 
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            pred.append(outputs.cpu().numpy())
            features.append(feature.cpu().numpy())
        features = np.concatenate(features)
        return np.concatenate(y_pred), np.concatenate(y_true), np.concatenate(pred)  
    
    def _evaluate(self, y_pred, y_true):
        ret = {}
        acc_total,grouped = accuracy(y_pred.T[0], y_true, self._known_classes,self.class_increments)
        return acc_total,grouped 
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
    def update_ema_variables(self, rapid_learner, conservative_learner, alpha=0.99, iter_num=0):
        for (rapid_param_name, rapid_param), (conservative_param_name, conservative_param) in zip(rapid_learner.named_parameters(), conservative_learner.named_parameters()):
            if "adapt" in rapid_param_name:
                conservative_param.data.mul_(alpha).add_(1 - alpha, rapid_param.data)
                print(f"update {conservative_param_name}")

    def update_ema_adapters(self, rapid_learner, conservative_learner, cur_task, alpha=0.99):
        rapid_state_dict = dict(rapid_learner.named_parameters())
        conservative_state_dict = dict(conservative_learner.named_parameters())
        latest_adapter_id = str(cur_task)

        for rapid_name, rapid_param in rapid_state_dict.items():
            if f"adaptmlp.adapters.{latest_adapter_id}." not in rapid_name:
                continue  
            conservative_name = rapid_name.replace(f"adapters.{latest_adapter_id}", "adapters.0")

            if conservative_name in conservative_state_dict:
                cons_param = conservative_state_dict[conservative_name]
                cons_param.data.mul_(alpha).add_(rapid_param.data, alpha=1 - alpha)
                print(f"use {rapid_name} to update {conservative_name}")
                
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self._batch_size= args["batch_size"]
        self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args=args

    def after_task(self):
        self._known_classes = self._classes_seen_so_far
    
    def replace_fc(self,trainloader):
        self._network = self._network.eval()

        if self.args['use_RP']:
            self._network.fc.use_RP=True
            if self.args['M']>0:
                self._network.fc.W_rand=self.W_rand
            else:
                self._network.fc.W_rand=None

        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.cuda()
                label=label.cuda()
                embedding = self._network.convnet(data)     
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        Y=target2onehot(label_list,self.total_classnum)
        if self.args['use_RP']:
            #print('Number of pre-trained feature dimensions = ',Features_f.shape[-1])
            if self.args['M']>0:
                Features_h=torch.nn.functional.relu(Features_f@ self._network.fc.W_rand.cpu())
            else:
                Features_h=Features_f
            self.Q=self.Q+Features_h.T @ Y 
            self.G=self.G+Features_h.T @ Features_h
            ridge=self.optimise_ridge_parameter(Features_h,Y)
            Wo=torch.linalg.solve(self.G+ridge*torch.eye(self.G.size(dim=0)),self.Q).T #better nmerical stability than .inv
            self._network.fc.weight.data=Wo[0:self._network.fc.weight.shape[0],:].to(device='cuda')
        else:
            for class_index in np.unique(self.train_dataset.labels):
                data_index=(label_list==class_index).nonzero().squeeze(-1)
                if self.is_dil:
                    class_prototype=Features_f[data_index].sum(0)
                    self._network.fc.weight.data[class_index]+=class_prototype.to(device='cuda') #for dil, we update all classes in all tasks
                else:
                    class_prototype=Features_f[data_index].mean(0)
                    self._network.fc.weight.data[class_index]=class_prototype

    def replace_fc_branch(self,trainloader):
        self.model_branch1 = self.model_branch1.eval()

        if self.args['use_RP']:
            self.model_branch1.fc.use_RP=True
            if self.args['M']>0:
                self.model_branch1.fc.W_rand=self.W_rand
            else:
                self.model_branch1.fc.W_rand=None

        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.cuda()
                label=label.cuda()
                embedding = self.model_branch1.convnet(data)     
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        Y=target2onehot(label_list,self.total_classnum)

        if self.args['use_RP']:
            if self.args['M']>0:
                Features_h=torch.nn.functional.relu(Features_f@ self.model_branch1.fc.W_rand.cpu())
            else:
                Features_h=Features_f
            self.Q=self.Q+Features_h.T @ Y 
            self.G=self.G+Features_h.T @ Features_h
            ridge=self.optimise_ridge_parameter(Features_h,Y)
            Wo=torch.linalg.solve(self.G+ridge*torch.eye(self.G.size(dim=0)),self.Q).T #better nmerical stability than .inv
            self.model_branch1.fc.weight.data=Wo[0:self.model_branch1.fc.weight.shape[0],:].to(device='cuda')
        else:
            for class_index in np.unique(self.train_dataset.labels):
                data_index=(label_list==class_index).nonzero().squeeze(-1)
                if self.is_dil:
                    class_prototype=Features_f[data_index].sum(0)
                    self.model_branch1.fc.weight.data[class_index]+=class_prototype.to(device='cuda') #for dil, we update all classes in all tasks
                else:
                    #original cosine similarity approach of Zhou et al (2023)
                    class_prototype=Features_f[data_index].mean(0)
                    self.model_branch1.fc.weight.data[class_index]=class_prototype

    def optimise_ridge_parameter(self,Features,Y):
        ridges=10.0**np.arange(3,9)
        num_val_samples=int(Features.shape[0]*0.8)
        losses=[]
        Q_val=Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:]
        G_val=Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:]
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T #better nmerical stability than .inv
            Y_train_pred=Features[num_val_samples::,:]@Wo.T
            losses.append(F.mse_loss(Y_train_pred,Y[num_val_samples::,:]))
            ridge=ridges[np.argmin(np.array(losses))]
        return ridge
    
    def incremental_train(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum()
        self._cur_task += 1
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        self._network.set_task_id(self._cur_task)

        if self._cur_task == 0:
            for block in self._network.convnet.blocks:
                if hasattr(block, "adaptmlp"):
                    block.adaptmlp.add_adapter()
        if self._cur_task > 0:
            for block_idx, block in enumerate(self._network.convnet.blocks):
                if hasattr(block, "adaptmlp"):
                    # Get the corresponding conservative adapter
                    source_adapter = self.model_branch1.convnet.blocks[block_idx].adaptmlp.adapters[0]
                    # Add the adapter to rapid learner, initialized from conservative adapter
                    block.adaptmlp.add_adapter(source_adapter=source_adapter)
    
        if self._cur_task > 0 and self.args['use_RP'] and self.args['M']>0:
            self._network.fc.weight.data = copy.deepcopy(self.train_fc).to(device='cuda')
            self._network.update_fc(self._classes_seen_so_far)
        else:
            self._network.update_fc(self._classes_seen_so_far) #creates a new head with a new number of classes (if CIL)
        if self.is_dil == False:
            logging.info("Starting CIL Task {}".format(self._cur_task+1))
        logging.info("Learning on classes {}-{}".format(self._known_classes, self._classes_seen_so_far-1))
        self.class_increments.append([self._known_classes, self._classes_seen_so_far-1])
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train", mode="train", ) 
        self.train_loader = DataLoader(self.train_dataset, batch_size=int(self._batch_size), shuffle=True, num_workers=num_workers)
        train_dataset_for_CPs = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train", mode="test", )
        self.train_loader_for_CPs = DataLoader(train_dataset_for_CPs, batch_size=self._batch_size, shuffle=True, num_workers=num_workers) 
        test_dataset = data_manager.get_dataset(np.arange(0, self._classes_seen_so_far), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=num_workers)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_CPs)

    def freeze_backbone(self,is_first_session=False):
        # Freeze the parameters for ViT.
        if 'vit' in self.args['convnet_type']:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "head." not in name and "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False
        else:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False

    def show_num_params(self,verbose=False):
        # show total parameters and trainable parameters
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} training parameters.')
        if total_params != total_trainable_params and verbose:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())


    def _train(self, train_loader, test_loader, train_loader_for_CPs):
        self._network.to(self._device)
    
        if self._cur_task == 0:
            args_ptm = {}
            args_ptm['convnet_type'] = self.args['convnet_type'].rpartition("_")[0] # PEFT is not used
            self.ptm = SimpleVitNet(args_ptm, True).to(self._device)
            self.ptm.eval()
        if self._cur_task > 0:
            if self.args['merge_result']:
                self.model_branch1.fc.weight.data = copy.deepcopy(self.train_fc_branch).to(device='cuda')
                self.model_branch1.update_fc(self._classes_seen_so_far)
                self.model_branch1.eval()
        
        if self._cur_task == 0 and self.dil_init==False:
            self.show_num_params()
            optimizer = optim.SGD([{'params':self._network.parameters()}], momentum=0.9, lr=self.args['body_lr'],weight_decay=self.weight_decay)
            scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
            #train the PETL method for the first task:
            logging.info("Starting PETL training on first task using "+self.args["model_name"]+" method")
            self._init_train(train_loader, test_loader, optimizer, scheduler)

            if self.args['merge_result']:
                self.model_branch1 = copy.deepcopy(self._network).to(self._device)

            if self.args['use_RP'] and self.dil_init==False:
                self.setup_RP() 
                if self.args['merge_result']:
                    self.setup_RP_branch()
                
        elif self._cur_task > 0 and self.dil_init==False:
            if self.args['follow_epoch']:
                self.show_num_params()
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.args['body_lr'],weight_decay=self.weight_decay)
                scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
                logging.info("Starting PETL training on first task using "+self.args["model_name"]+" method")
                self._follow_train(train_loader, test_loader, optimizer, scheduler)
                self.update_ema_adapters(self._network, self.model_branch1, self._cur_task, alpha=0.99)

            if self.args['use_RP'] and self.dil_init==False:
                self.setup_RP_follow() 
                if self.args['merge_result']:
                    self.setup_RP_follow_branch()

        if self.is_dil and self.dil_init==False:
            self.dil_init=True
            self._network.fc.weight.data.fill_(0.0)
        
        self.replace_fc(train_loader_for_CPs)
        if self.args['merge_result']:
            self.replace_fc_branch(train_loader_for_CPs)
        self.show_num_params()
    
    def setup_RP_branch(self):
        self.initiated_G=False
        self.model_branch1.fc.use_RP=True
        if self.args['M']>0:
            #RP with M > 0
            M=self.args['M']
            self.train_fc_branch = copy.deepcopy(self.model_branch1.fc.weight)
            self.model_branch1.fc.weight = nn.Parameter(torch.Tensor(self.model_branch1.fc.out_features, M).to(device='cuda')) #num classes in task x M
            self.model_branch1.fc.reset_parameters()
            self.model_branch1.fc.W_rand=torch.randn(self.model_branch1.fc.in_features,M).to(device='cuda')
            self.W_rand_branch=copy.deepcopy(self.model_branch1.fc.W_rand) #make a copy that gets passed each time the head is replaced
        else:
            #no RP, only decorrelation
            M=self.model_branch1.fc.in_features #this M is L in the paper
        self.Q_branch=torch.zeros(M,self.total_classnum)
        self.G_branch=torch.zeros(M,M)

    def setup_RP(self):
        self.initiated_G=False
        self._network.fc.use_RP=True
        if self.args['M']>0:
            #RP with M > 0
            M=self.args['M']
            self.train_fc = copy.deepcopy(self._network.fc.weight)
            self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).to(device='cuda')) #num classes in task x M
            self._network.fc.reset_parameters()
            self._network.fc.W_rand=torch.randn(self._network.fc.in_features,M).to(device='cuda')
            self.W_rand=copy.deepcopy(self._network.fc.W_rand) #make a copy that gets passed each time the head is replaced
        else:
            #no RP, only decorrelation
            M=self._network.fc.in_features #this M is L in the paper
        self.Q=torch.zeros(M,self.total_classnum)
        self.G=torch.zeros(M,M)

    def setup_RP_follow(self):
        self.train_fc = copy.deepcopy(self._network.fc.weight)

    def setup_RP_follow_branch(self):
        self.train_fc_branch = copy.deepcopy(self.model_branch1.fc.weight)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(int(self.args['tuned_epoch'])))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                losses += loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss_ce {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _follow_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(int(self.args['follow_epoch'])))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses_ce, losses_rad = 0.0, 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                if self.args['merge_result']:
                    loss_fa, loss_CR = self.radical(inputs, targets)
                    loss_radical = loss_CR
                    loss += loss_radical
                    losses_rad += loss_radical

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses_ce += loss

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
          
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss_ce {:.3f}, Loss_rad {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                int(self.args['follow_epoch']),
                losses_ce / len(train_loader),
                losses_rad / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)
        
    def radical(self, inputs, targets):
        features_s = self._network.convnet(inputs)

        with torch.no_grad():
            self.model_branch1.eval()
            features_t = self.model_branch1.convnet(inputs)

        s = F.cosine_similarity(features_s,features_t, dim=-1)
        loss_fa = torch.sum(1 - s)

        f_bcl_ptm = torch.cat([self.model_branch1.fc.weight[:self._known_classes], features_t], dim=0)
        targets_bcl = torch.cat([torch.arange(self._known_classes).to(self._device), targets.to(self._device)], dim=0)
        
        f_bcl_cur = torch.cat([self._network.fc.weight[:self._known_classes], features_s], dim=0) 

        pred_bcl_ptm = self._network.fc(f_bcl_ptm)["logits"]
        pred_bcl_cur = self.model_branch1.fc(f_bcl_cur)["logits"]
        loss_CR = F.cross_entropy(pred_bcl_cur, targets_bcl) 

        return loss_fa, loss_CR
    
