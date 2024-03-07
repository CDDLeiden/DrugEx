from drugex.data.corpus.vocabulary import VocGraph
from drugex.data.datasets import GraphFragDataSet
from drugex.training.models import GraphTransformer
from drugex.training.monitors import FileMonitor
from ..settings import *

def main():
    data_sets = [GraphFragDataSet(f'{OUTPUT_FILE}_{split}.tsv', rewrite=False) for split in ('test', 'train')]
    # voc = VocSmiles.fromFile(f'{OUTPUT_FILE}_train.tsv.vocab', True)
    voc = VocGraph.fromFile(f'{OUTPUT_FILE}_train.tsv.vocab')
    print('Training')
    agent = GraphTransformer(voc, use_gpus=GPUS)
    monitor = FileMonitor(OUTPUT_FILE, verbose=True)
    agent.fit(data_sets[1].asDataLoader(BATCH_SIZE), data_sets[0].asDataLoader(BATCH_SIZE), epochs=N_EPOCHS, monitor=monitor)
    print('Training done.')

if __name__ == '__main__':
    main()
