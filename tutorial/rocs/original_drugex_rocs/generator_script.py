import os
from drugex.training.generators import SequenceRNN
from drugex.data.corpus.vocabulary import VocSmiles

MODELS_PR_PATH = "/home/sem23/fastrocs/rp1-sem-egbers/DrugEx_script/RL_ns10000_eps0.01_bs128_pt1000_ne1000/"
GPUS = [0] #we will use only one GPU with ID=0, but if you have more, you can list more GPU IDs here

voc = VocSmiles.fromFile(os.path.join('/home/sem23/fastrocs/rp1-sem-egbers/DrugEx_script/pretrained.vocab'), encode_frags=False)
pretrained = SequenceRNN(voc, is_lstm=True, use_gpus=GPUS)
pretrained.loadStatesFromFile(os.path.join(MODELS_PR_PATH, "CCR2_agent.pkg"))

generated = pretrained.generate(num_samples=1000, batch_size=128)
generated.to_csv(os.path.join(MODELS_PR_PATH,'generated_10.tsv'))