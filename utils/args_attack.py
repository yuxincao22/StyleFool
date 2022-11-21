import argparse

def video_attack_args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int, required=True, help='The gpus to use')
    parser.add_argument('--untargeted', action='store_true')
    parser.add_argument('--video', type=str, default='orig.npy')
    parser.add_argument('--label', type=int, default=0)
    parser.add_argument('--target-video', type=str, default='target.npy')
    parser.add_argument('--target-label', type=int, default=10)
    parser.add_argument('--adv-save-path', type=str, default='adv.npy')
    parser.add_argument('--no_rank_transform', action='store_true')
    parser.add_argument('--sigma', type=float, default=1e-6)
    parser.add_argument('--sample_per_draw', type=int, default=48, help='Number of samples used for NES')
    parser.add_argument('--sub_num_sample', type=int, default=12,nhelp='Number of samples processed each time')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = video_attack_args_parse()
    print(args)
