import argparse


class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        # Model related arguments
        parser.add_argument('--num_mix', default=2, type=int,
                            help="number of sounds to mix")
        parser.add_argument('--batch', default=0,
                            help="architecture of net_frame")
        parser.add_argument('--save_per_batchs', default=10,
                            help="the model, and pictures are save every this value")
        parser.add_argument('--out_threshold', default=0.5, type=int,
                            help='threshold apply on the predicted masks')

        # Data related arguments
        parser.add_argument('--audLen', default=24000, type=int,
                            help='sound length')
        parser.add_argument('--audRate', default=48000, type=int,
                            help='sound sampling rate')
        parser.add_argument('--stft_frame', default=1022, type=int,
                            help="stft frame length")
        parser.add_argument('--stft_hop', default=256, type=int,
                            help="stft hop length")
        parser.add_argument('--beta1', default=0.9, type=float,
                            help='momentum for sgd, beta1 for adam')
        parser.add_argument('--mode', default='train', type=str,
                            help='weights regularizer')
        parser.add_argument('--path', default='', type=str,
                            help='configure the path where to save data')
        parser.add_argument('--augment', default='', type=str,
                            help='type of data augmentation applied') 
        parser.add_argument('--species', default='', type=list,
                            help='species training dataset')    
        parser.add_argument('--name_classes', default='', type=str,
                            help='configure the name of the classes')

        self.parser = parser

    def add_train_arguments(self):
        parser = self.parser

        # optimization related arguments
        parser.add_argument('--lr_sound', default=1e-3, type=float, help='LR')
        self.parser = parser

#    def print_arguments(self, args):
#        resume = open('resume.txt', 'a')
#        print("Input arguments:")
#        for key, val in vars(args).items():
#            resume.writelines([key, str(val), '\n'])
#            print("{:16} {}".format(key, val))
#        resume.close()

    def parse_train_arguments(self):
        self.add_train_arguments()
        args = self.parser.parse_args()
#        self.print_arguments(args)
        return args
