from data_utils import HCPAScFcDataset, ATLAS_FACTORY, HCPAScFcDatasetOnDisk, PE_K, MDNN_MAX_DEGREE
# from models import ToyMPNN
import models
# from torch_geometric.nn import GINConv, GCNConv, GATConv
import sys
import torch_geometric
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from tqdm import trange, tqdm
# from sklearn.metrics import 

torch.manual_seed(142857)

def main():
    dloader_num_workers = 16
    print("Command: python", ' '.join(list(sys.argv)))
    # nn_type = 'mpnn' if sys.argv[7].lower() not in ['mpnn', 'mdnn'] else sys.argv[7].lower()
    # node_attr = sys.argv[4].upper()
    # atlas = sys.argv[3]
    # do_retest = sys.argv[6]=="1"
    nn_type = 'mdnn'
    node_attr = 'FC'
    atlas = 'AAL_116'
    do_retest = False
    nlayer = 4
    device = 'cuda:3'
    use_skip = True
    gnn_type = 'GINConv'
    nlayer = 4
    heads = 4
    wo_dee = True
    assert atlas in ATLAS_FACTORY, f"{atlas} not in {ATLAS_FACTORY}"
    if not do_retest:
        dset = HCPAScFcDatasetOnDisk(atlas, node_attr=node_attr, nn_type=nn_type)
    else:
        dset = HCPAScFcDatasetOnDisk(atlas, node_attr=node_attr, direct_filter=['PA'], nn_type=nn_type)
        retest_dset = HCPAScFcDatasetOnDisk(atlas, node_attr=node_attr, direct_filter=['AP'], nn_type=nn_type)
    print('Loaded Dataset')
    if node_attr=='FC':
        inch = list(dset.all_sc.values())[0].shape[-1]
    elif node_attr=='BOLD':
        inch = dset.fc_winsize
    elif node_attr=='SC':
        inch = list(dset.all_sc.values())[0].shape[-1]
    elif node_attr=='ID':
        inch = 1
    elif node_attr=='DEN':
        inch = 1
    elif node_attr=='DE':
        inch = list(dset.all_sc.values())[0].shape[-1]
    elif node_attr=='FC+DE':
        inch = list(dset.all_sc.values())[0].shape[-1]*2
    elif node_attr=='SC+DE':
        inch = list(dset.all_sc.values())[0].shape[-1]*2
    else:
        print(f"{node_attr} not support")
        exit()

    # nlayer = int(sys.argv[5])
    hidch = 768
    bs = 32 if nlayer <= 4 else 16
    lr = 0.001 if nlayer <= 4 else 0.00005
    num_epochs = 100
    max_patience = 50
    # device = sys.argv[-1]
    # use_skip = sys.argv[2]=='1'
    # heads = 4
    # Get all subjects from the dataset
    all_subjects = dset.data_subj

    # Define 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=142857)

    # Initialize lists to store evaluation metrics
    accuracies = []
    f1_scores = []
    retest_accuracies = []
    prec_scores = []
    rec_scores = []
    # retest_f1_scores = []
    # finetune_ratio = 0.5
    few_shot_num = torch.inf
    for fold, (train_index, val_index) in enumerate(kf.split(all_subjects)):
        # if fold != 1:
        #     continue
        model = getattr(models, f'Toy{nn_type.upper()}')(getattr(torch_geometric.nn, gnn_type), nlayer, inch, len(dset.fc_task.unique()), hidch, is_graph_level=True, 
                                                         pedim=PE_K*2, trans_nhead=2, trans_nlayer=2, wo_dee=wo_dee)

        # model.pe_lin = nn.Identity()
        # model.lin_identifier = nn.Identity()
        # model.transformer = models.DetourTransformerEncoder(nlayer=nlayer, in_channels=2*inch+1, out_channels=hidch, heads=heads, pedim=PE_K*2)
        # # model.transformer.load_state_dict(torch.load(f'detour_encoder_l4h4_hcpa_alldataFC{node_attr}_best.pt'))
        # model.transformer.lin_out = models.TopkEncoder(MDNN_MAX_DEGREE, hidch)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Get the subjects for training and validation from the split
        train_subjects = [all_subjects[i] for i in train_index]
        val_subjects = [all_subjects[i] for i in val_index]

        # Filter dataset based on training and validation subjects
        # train_data = [data for data in dset if data['subject'] in train_subjects]
        # val_data = [data for data in dset if data['subject'] in val_subjects]
        train_data = [di for di, subj in enumerate(dset.fc_subject) if subj in train_subjects]
        val_data = [di for di, subj in enumerate(dset.fc_subject) if subj in val_subjects]
        print(f'Fold {fold + 1}, Train {len(train_subjects)} subjects, Val {len(val_subjects)} subjects, len(train_data)={len(train_data)}, len(val_data)={len(val_data)}')
        train_data = torch.utils.data.Subset(dset, train_data)
        val_data = torch.utils.data.Subset(dset, val_data)
        
        trainloader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=dloader_num_workers)
        valloader = DataLoader(val_data, batch_size=bs, shuffle=False, num_workers=dloader_num_workers)
        if do_retest:
            retestloader = DataLoader(retest_dset, batch_size=bs, shuffle=False, num_workers=dloader_num_workers)

        best_f1 = 0
        patience = 0
        # Train the model
        for epoch in (pbar := trange(num_epochs, desc='Epoch')):
            model.train()
            # for batch in tqdm(trainloader, desc='Training'):
            bi = 0
            shoti = [0 for _ in range(4)]
            for batch in trainloader:
                # if bi > finetune_ratio*len(trainloader): break
                bi += 1
                inputs, labels = batch['data'].to(device), batch['label'].to(device)
                trainmask = []
                for di, y in enumerate(labels):
                    y = y.cpu().item()
                    if shoti[y] > few_shot_num: continue
                    shoti[y] += 1
                    trainmask.append(di)
                if len(trainmask) == 0: continue
                optimizer.zero_grad()
                if nn_type == 'mpnn':
                    outputs = model(inputs.x, inputs.edge_index, batch=inputs.batch, skip_connect=use_skip)
                elif nn_type == 'mdnn':
                    outputs = model(inputs.x, inputs.edge_index, de_x=[inputs.node_attr, inputs.pe, inputs.dee, inputs.id, inputs.pad_mask], batch=inputs.batch, skip_connect=use_skip)
                loss = criterion(outputs[trainmask], labels[trainmask])
                loss.backward()
                optimizer.step()

            # Validate the model
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in valloader:
                    inputs, labels = batch['data'].to(device), batch['label'].to(device)
                    if nn_type == 'mpnn':
                        outputs = model(inputs.x, inputs.edge_index, batch=inputs.batch, skip_connect=use_skip)
                    elif nn_type == 'mdnn':
                        outputs = model(inputs.x, inputs.edge_index, de_x=[inputs.node_attr, inputs.pe, inputs.dee, inputs.id, inputs.pad_mask], batch=inputs.batch, skip_connect=use_skip)                    
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            # Calculate evaluation metrics
            acc = accuracy_score(all_labels, all_preds)
            # f1 = f1_score(all_labels, all_preds, average='weighted')
            
            prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
            # print(prec, rec, f1)
            if do_retest:
                
                # Re-test the model
                model.eval()
                retest_preds = []
                retest_labels = []
                with torch.no_grad():
                    for batch in retestloader:
                        inputs, labels = batch['data'].to(device), batch['label'].to(device)
                        if nn_type == 'mpnn':
                            outputs = model(inputs.x, inputs.edge_index, batch=inputs.batch, skip_connect=use_skip)
                        elif nn_type == 'mdnn':
                            outputs = model(inputs.x, inputs.edge_index, de_x=[inputs.node_attr, inputs.pe, inputs.dee, inputs.id, inputs.pad_mask], batch=inputs.batch, skip_connect=use_skip)                        
                        _, preds = torch.max(outputs, 1)
                        retest_preds.extend(preds.cpu().numpy())
                        retest_labels.extend(labels.cpu().numpy())
                # Calculate evaluation metrics
                retest_acc = accuracy_score(retest_labels, retest_preds)
                # retest_f1 = f1_score(retest_labels, retest_preds, average='weighted')

            if do_retest:
                pbar.set_description(f'Accuracy: {acc}, F1 Score: {f1}, Retest-Accuracy: {retest_acc}, Epoch')
            else:
                pbar.set_description(f'Accuracy: {acc}, F1 Score: {f1}, Epoch')
            if f1 >= best_f1:
                if f1 > best_f1: 
                    patience = 0
                else:
                    patience += 1
                best_f1 = f1
                best_acc = acc
                best_prec = prec
                best_rec = rec
                if do_retest:
                    # best_re_f1 = retest_f1
                    best_re_acc = retest_acc
            else:
                patience += 1
            if patience > max_patience: break
        accuracies.append(best_acc)
        f1_scores.append(best_f1)
        prec_scores.append(best_prec)
        rec_scores.append(best_rec)
        if do_retest:
            retest_accuracies.append(best_re_acc)
            # retest_f1_scores.append(best_re_f1)

        print(f'Accuracy: {best_acc}, F1 Score: {best_f1}, Prec: {best_prec}, Rec: {best_rec}')

    # Calculate mean and standard deviation of evaluation metrics
    mean_accuracy = sum(accuracies) / len(accuracies)
    std_accuracy = torch.std(torch.tensor(accuracies))
    mean_f1_score = sum(f1_scores) / len(f1_scores)
    std_f1_score = torch.std(torch.tensor(f1_scores))
    mean_prec_score = sum(prec_scores) / len(prec_scores)
    std_prec_score = torch.std(torch.tensor(prec_scores))
    mean_rec_score = sum(rec_scores) / len(rec_scores)
    std_rec_score = torch.std(torch.tensor(rec_scores))

    print(f'Mean Accuracy: {mean_accuracy}, Std Accuracy: {std_accuracy}')
    print(f'Mean F1 Score: {mean_f1_score}, Std F1 Score: {std_f1_score}')
    print(f'Mean prec Score: {mean_prec_score}, Std prec Score: {std_prec_score}')
    print(f'Mean rec Score: {mean_rec_score}, Std rec Score: {std_rec_score}')
    if do_retest:
        print(f'Mean Retest-Accuracy: {sum(retest_accuracies) / len(retest_accuracies)}, Std Accuracy: {torch.std(torch.tensor(retest_accuracies))}')
        # print(f'Mean Retest-F1 Score: {sum(retest_f1_scores) / len(retest_f1_scores)}, Std F1 Score: {torch.std(torch.tensor(retest_f1_scores))}')

if __name__ == '__main__': main()