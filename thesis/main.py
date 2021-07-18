from ctgan.synthesizers.ctgan import CTGANSynthesizer
from reader import get_datasets, read_csv, write_csv
import time
import warnings
#warnings.filterwarnings('ignore')

datasets = get_datasets()
discriminator_steps = 5
verbose = True

for dataset in datasets:

    data = read_csv(dataset)
    print('Dataset name: ' + dataset + ', Rows: ' + str(len(data)) + ', Features: ' + str(len(data.columns)))

    for early_stop in [0, 1]:

        # Learning:
        synthesizer = CTGANSynthesizer(epochs=10, verbose=verbose, discriminator_steps = discriminator_steps)
        print('CTGAN initialized')

        print('Fit started')
        start_time = time.time()
        synthesizer.fit(data, data.select_dtypes('object').columns, early_stop=early_stop, dataset_name=dataset.replace(".csv", ""))

        print('Dataset name: ' + dataset + ', Time: ' + str(time.time() - start_time))

        # Synthesis
        sample = synthesizer.sample(len(data))
        print('Dataset generated')

        # Save
        write_csv(sample, dataset, early_stop)
        print('Dataset saved')

