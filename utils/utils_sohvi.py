import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

import utils
from models import generator, GPT2Model, GraphModel
from models.explorer import SmilesExplorer, GraphExplorer

def DataPreparationGraph(args):
    
    data_path = args.base_dir + '/data/'

    voc = utils.VocGraph( data_path + 'voc_graph.txt', max_len=80, n_frags=4)
    
    data = pd.read_table( data_path + '%s_train_code.txt' % args.input)
    data = torch.from_numpy(data.values).long().view(len(data), voc.max_len, -1)
    train_loader = DataLoader(data, batch_size=args.batch_size * 4, drop_last=True, shuffle=True)

    test = pd.read_table( data_path + '%s_test_code.txt' % args.input)
    # test = test.sample(int(1e4))
    test = torch.from_numpy(test.values).long().view(len(test), voc.max_len, -1)
    valid_loader = DataLoader(test, batch_size=args.batch_size * 10, drop_last=True, shuffle=True)
    
    return voc, train_loader, valid_loader

def DataPreparationSmiles(args):
    
    data_path = args.base_dir + '/data/'
    
    if args.method in ['gpt']:
        voc = utils.Voc( data_path + 'voc_smiles.txt', src_len=100, trg_len=100)
    else:
        voc = utils.VocSmiles( data_path + 'voc_smiles.txt', max_len=100)

    data = pd.read_table( data_path + '%s_train_smi.txt' % args.input)
    data_in = voc.encode([seq.split(' ') for seq in data.Input.values])
    data_out = voc.encode([seq.split(' ') for seq in data.Output.values])
    data_set = TensorDataset(data_in, data_out)
    train_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True)
    

    test = pd.read_table( data_path + '%s_test_smi.txt' % args.input)
    print(len(test), 10*args.batch_size)
    test = test.Input.drop_duplicates()
    print(len(test))
    #test = test.sample(args.batch_size * 10).values
    test_set = voc.encode([seq.split(' ') for seq in test])
    test_set = utils.TgtData(test_set, ix=[voc.decode(seq, is_tk=False) for seq in test_set])
    valid_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=test_set.collate_fn)
    
    return voc, train_loader, valid_loader


def SetAlgorithm(voc, alg, method='gpt'):
    
    if alg == 'smiles':
        if method == 'ved':
            agent = generator.EncDec(voc, voc).to(utils.dev)
        elif method == 'attn':
            agent = generator.Seq2Seq(voc, voc).to(utils.dev)
        elif method == 'gpt':
            agent = GPT2Model(voc, n_layer=12).to(utils.dev)
    elif alg == 'graph':
        agent = GraphModel(voc).to(utils.dev)
    
    return agent

def InitializeEvolver(agent, prior, args):
    
    if args.algorithm == 'smiles':
        evolver = SmilesExplorer(agent, mutate=prior)
    elif args.algorithm == 'graph':
        evolver = GraphExplorer(agent, mutate=prior)
        
    evolver.batch_size = args.batch_size 
    evolver.epsilon = args.epsilon 
    evolver.sigma = args.beta 
    evolver.scheme = args.scheme 
    evolver.repeat = 1   
    
    return evolver
    

def CreateDesirabilityFunction(args):
    
    objs, keys = [], []

    for t in args.targets:
        if args.env_alg.startswith('MT_'):
            sys.exit('TO DO: using multitask model')
        else:
            path = args.base_dir + '/envs/single/' + args.env_alg + '_' + args.env_task + '_' + t + '.pkg'
        objs.append(utils.Predictor(path, type=args.env_task))
        keys.append(t)
    objs.append(utils.Property('QED'))
    keys.append('QED')
    
    return objs, keys 

def SetModes(scheme, keys, env_task, active_targets, inactive_targets):
    
    """ Calculate clipped scores and threasholds for each task (targets andd QED) depending the evolver scheme ('WS' or other) and environement taks ('REG' or 'CLS')"""
    
    if scheme == 'WS':
        if env_task == 'CLS':
            active = utils.ClippedScore(lower_x=0.2, upper_x=0.5)
            inactive = utils.ClippedScore(lower_x=0.5, upper_x=0.8)
        else:
            active = utils.ClippedScore(lower_x=3, upper_x=10)
            inactive = utils.ClippedScore(lower_x=10, upper_x=3)
        qed = utils.ClippedScore(lower_x=0, upper_x=1)
        ths = [0.5] * (len(active_targets)+len(inactive_targets)) + [0.0]
        print(ths)
    else:
        if env_task == 'CLS':
            active = utils.ClippedScore(lower_x=0.2, upper_x=0.5)
            inactive = utils.ClippedScore(lower_x=0.5, upper_x=0.8)
        else:
            active = utils.ClippedScore(lower_x=3, upper_x=6.5)
            inactive = utils.ClippedScore(lower_x=10, upper_x=6.5)
        qed = utils.ClippedScore(lower_x=0, upper_x=0.5)
        ths = [0.99] * (len(active_targets)+len(inactive_targets)) + [0.0]
        print(ths)            
        
    mods = []
    for k in keys:
        if k in active_targets: 
            mods.append(active)
        elif k in inactive_targets: 
            mods.append(inactive)
        elif k == 'QED' : 
            mods.append(qed)
    
    return mods, ths