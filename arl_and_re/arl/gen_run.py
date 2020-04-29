import argparse
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--output", default='job.sh', type=str,
                    help="output sh file name")
parser.add_argument("--gpuid", default=0, type=int,
                    help="output sh file name")

args = parser.parse_args()
output_name = args.output
save = open(output_name, 'w')
save.write('#!/bin/bash' + '\n')



source_language = 'en'
gpu = args.gpuid

for rndseed in [0]:
    for lr in [3e-5, 5e-5, 7e-5, 9e-5]:
        for batchsize in [32]:
            for max_epoch in [3, 5, 7]:
                for warmup in [0.4]:

                    script = "python main_bibert.py --exp_name gpu%d --gpuid %d --batchsize %d --warmup_proportion %.5f" \
                         " --learning_rate %.6f --max_epoch %d"  % \
                         (gpu, gpu, batchsize, warmup, lr, max_epoch)

                    save.write(script + '\n')

save.close()