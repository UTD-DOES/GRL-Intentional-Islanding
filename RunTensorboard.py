from tensorboard import default
from tensorboard import program


path_add = "Tensorboard_logger/PPO_84"

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None,'--logdir',path_add])
    tb.main()
    
