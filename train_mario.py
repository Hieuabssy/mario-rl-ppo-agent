import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from ppo import PPO

# ============================================================================
# Select level to train - CHANGE HERE
# ============================================================================
WORLD = 1  # Select world (1-8)
STAGE = 2  # Select stage (1-4)

TOTAL_TIMESTEPS = 1000000  # total training steps
CHECKPOINT_FREQ = 100000    # Save checkpoint every 100,000 steps
LEARNING_RATE = 0.00001   # Learning rate
N_STEPS = 512              # Number of steps per update

STAGE_NAME = f"{WORLD}-{STAGE}"
CHECKPOINT_DIR = f"./train/{STAGE_NAME}/"
LOG_DIR = f"./logs/{STAGE_NAME}/"
VIDEO_DIR = f"./videos/{STAGE_NAME}/"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

print(f"TRAINING AI FOR SUPER MARIO BROS - LEVEL {STAGE_NAME}")
print(f"Checkpoint: {CHECKPOINT_DIR}")
print(f"Logs: {LOG_DIR}")
print(f"Videos: {VIDEO_DIR}")

class BaseCallback:
    def __init__(self, verbose=0):
        self.n_calls = 0
        self.model = None
        self.verbose = verbose

    def _init_callback(self):
        pass

    def _on_step(self):
        return True

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(
                self.save_path, 
                f'best_model_{self.n_calls}.pth'
            )
            self.model.save(model_path)
            print(f"Saved checkpoint at step {self.n_calls}")
        return True

# ============================================================================
# CUSTOM REWARD CONFIGURATION (MUST MATCH TRAIN)
# ============================================================================
USE_CUSTOM_REWARD = False  # Set to False if not using custom reward

if USE_CUSTOM_REWARD:
    try:
        from custom_rewards import CustomRewardWrapper
        print("Custom reward wrapper imported!")
    except ImportError:
        print("Not found custom_rewards.py, will use original reward")
        USE_CUSTOM_REWARD = False

# ============================================================================
# ENV CREATION FUNCTION
# ============================================================================
def create_mario_env(world, stage):
    env_name = f'SuperMarioBros-{world}-{stage}-v0'
    print(f"Creating environment: {env_name}")
    
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    if USE_CUSTOM_REWARD:
        env = CustomRewardWrapper(env)
        print("Custom reward wrapper applied!")
    
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    
    print("Environment created and preprocessed successfully!")
    return env

# ============================================================================
# Train model function
# ============================================================================
def train_model():
    """
    Train PPO model for Super Mario Bros
    """
    print("START TRAINING")
    
    env = create_mario_env(WORLD, STAGE)
    
    callback = TrainAndLoggingCallback(
        check_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR
    )
    callback._init_callback()
    
    print("Initializing PPO model...")
    model = PPO(
        'CnnPolicy',           
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS
    )
    print("Model initialized!")
    
    print(f"Start training with {TOTAL_TIMESTEPS} timesteps...")
    print(f"Checkpoint will be saved every {CHECKPOINT_FREQ} steps")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback
    )
    
    final_model_path = os.path.join(CHECKPOINT_DIR, f'final_model_{STAGE_NAME}.pth')
    model.save(final_model_path)
    
    print("TRAINING COMPLETED!")
    print(f"Final model saved at: {final_model_path}")

    env.close()
    return final_model_path

if __name__ == "__main__":
    try:
        final_model = train_model()
        print("Use test_mario.py to test the model!")
        print(f"Model path: {final_model}")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise