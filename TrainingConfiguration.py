import argparse
import torch

def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="Graph neural network based reinforcement learning for intentional islanding")
    
    parser.add_argument('--FeatureDim', type=int, default=128, help="Embedding length")
    #parser.add_argument('--TotalSteps', type=int, default=5000000, help='Total number of steps')
    parser.add_argument('--TotalSteps', type=int, default=5, help='Total number of steps')

    parser.add_argument('--BatchSize', type=int, default=2, help='Batch size for training')  
    parser.add_argument('--Nsteps', type=int, default=1500, help='Number of steps for rollout')
    parser.add_argument('--LearningRate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--EntropyCoef', type=float, default=0.0, help='Entropy coefficient')
    parser.add_argument('--ValueCoef', type=float, default=0.5, help='Value coefficient')
    parser.add_argument('--Gamma', type=float, default=1.00, help='Discount factor')
    parser.add_argument('--NEpochs', type=int, default=100, help='Number of epochs per rollout')

    parser.add_argument('--SaveFreq', type=int, default=5000, help="Save frequency")

    parser.add_argument('--Logger', type=str, default='Tensorboard_logger/', help='Directory for tensorboard logger')
    parser.add_argument('--ModelSave', type=str, default='Trained_Models/',
                        help='Directory for saving the trained models')
    parser.add_argument('--NoCUDA', action='store_true', help='Disable CUDA')
    parser.add_argument('--NumCPU', type=int, default=1, help="Number of parallel environments for rollout")

    config = parser.parse_args(args)
    config.UseCUDA = torch.cuda.is_available() and not config.NoCUDA

    return config
