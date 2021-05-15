ENV_NAME = 'BreakoutDeterministic-v0'

LOAD_FROM = None 
SAVE_PATH = 'breakout-saves'
LOAD_REPLAY_BUFFER = True

WRITE_TENSORBOARD = True
TENSORBOARD_DIR = 'tensorboard/'

USE_PER = False

PRIORITY_SCALE = 0.7             
CLIP_REWARD = True                


TOTAL_FRAMES = 3000000 #00         
MAX_EPISODE_LENGTH = 18000        
FRAMES_BETWEEN_EVAL = 100000     
EVAL_LENGTH = 10000               
UPDATE_FREQ = 10000               

DISCOUNT_FACTOR = 0.99          
MIN_REPLAY_BUFFER_SIZE = 20000 #50000    
MEM_SIZE = 1000000 #1000000                

MAX_NOOP_STEPS = 20               
UPDATE_FREQ = 4                 

INPUT_SHAPE = (84, 84)           
BATCH_SIZE = 8                 
LEARNING_RATE = 0.00001