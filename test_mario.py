import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from ppo import PPO
from gym.wrappers import RecordVideo
import glob

# ============================================================================
# Select level to test - CHANGE HERE
# ============================================================================
WORLD = 1  # Must match trained world
STAGE = 2  # Must match trained stage

RECORD_VIDEO = True         # Is recording video?
NUM_EPISODES = 25           # Number of episodes to test
RENDER_GAME = True          # Render the game during testing

# choose model to test or None to select interactively
MODEL_PATH = None  # ex: "./train/1-1/best_model_100000" or None

STAGE_NAME = f"{WORLD}-{STAGE}"
CHECKPOINT_DIR = f"./train/{STAGE_NAME}/"
VIDEO_DIR = f"./videos/{STAGE_NAME}/"

os.makedirs(VIDEO_DIR, exist_ok=True)

print(f"TEST AI FOR SUPER MARIO BROS - STAGE {STAGE_NAME}")

# ============================================================================
# CUSTOM REWARD MUST MATCH TRAIN
# ============================================================================
USE_CUSTOM_REWARD = False  # Use custom reward function?

if USE_CUSTOM_REWARD:
    try:
        from custom_rewards import CustomRewardWrapper
    except ImportError:
        print("Not found custom_rewards.py, will use original reward")
        USE_CUSTOM_REWARD = False

# ============================================================================
# CREATE MARIO ENV
# ============================================================================
def create_mario_env(world, stage, record_video=False, video_folder=None):
    """
    Create Super Mario Bros env for testing
    """
    env_name = f'SuperMarioBros-{world}-{stage}-v0'
    env = gym_super_mario_bros.make(env_name)
    
    if record_video and video_folder:
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda x: True,  
            name_prefix=f"mario_{STAGE_NAME}_test"
        )
    
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    if USE_CUSTOM_REWARD:
        env = CustomRewardWrapper(env)
    
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    
    return env

# ============================================================================
# FIND OR LIST MODELS
# ============================================================================
def find_latest_model(checkpoint_dir):
    model_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    
    if not model_files:
        raise FileNotFoundError(
            f"Not found model in {checkpoint_dir}\n"
            f"run train_mario.py before!"
        )
    
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model.replace('.pth', '')

def list_available_models(checkpoint_dir):
    model_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    
    if not model_files:
        return []
    
    models = []
    for f in sorted(model_files):
        name = os.path.basename(f).replace('.pth', '')
        size = os.path.getsize(f) / (1024 * 1024)  # MB
        models.append({
            'name': name,
            'path': f.replace('.pth', ''),
            'size_mb': round(size, 2)
        })
    
    return models

# ============================================================================
# TEST MODEL FUNCTION
# ============================================================================
def test_model(model_path, num_episodes=3, render=True):
    print(f"Loading model: {model_path}")
    
    try:
        video_folder = VIDEO_DIR if RECORD_VIDEO else None
        env = create_mario_env(WORLD, STAGE, RECORD_VIDEO, video_folder)
        model = PPO.load(model_path, env=env)
        print("Model load finish!")
    except Exception as e:
        print(f"Error when load model: {e}")
        return
    
    if RECORD_VIDEO:
        print(f"Video will be save at: {VIDEO_DIR}")
    
    print(f"Started playing {num_episodes} times...")
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        while not done:
            action, _ = model.predict(state)
            
            state, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            
            if render:
                env.render()
        
        print(f"   Completed later {steps} steps")
        print(f"   Total reward: {total_reward:.2f}")
        
        if info[0].get('flag_get'):
            print(f"   FLAG IS UP! Mario has completed the level!")
        else:
            print(f"   Mario is dead or out of time")
    
    print("TEST COMPLETE!")
    
    if RECORD_VIDEO:
        print(f"The video has been saved at: {VIDEO_DIR}")
    
    env.close()

# ============================================================================
# INTERACTIVE MODEL SELECTION
# ============================================================================
def interactive_model_selection():
    models = list_available_models(CHECKPOINT_DIR)
    
    if not models:
        print(f"No model found in {CHECKPOINT_DIR}")
        print("Run train_mario.py first!")
        return None
    
    print("AVAILABLE MODELS:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']} ({model['size_mb']} MB)")
    
    while True:
        try:
            choice = input(f"Select model (1-{len(models)}) or Enter to select latest: ").strip()
            
            if choice == "":
                return models[-1]['path']
            
            choice = int(choice)
            if 1 <= choice <= len(models):
                return models[choice - 1]['path']
            else:
                print(f"Please select a number from 1 to {len(models)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("Cancelled")
            return None

# ============================================================================
# RUN PROGRAM
# ============================================================================
if __name__ == "__main__":
    try:
        if MODEL_PATH is None:
            print("Looking for model...")
            model_path = interactive_model_selection()
            
            if model_path is None:
                print("No model to test. Exit...")
                exit(1)
        else:
            model_path = MODEL_PATH
            
            if not os.path.exists(model_path + '.pth'):
                print(f"Model does not exist: {model_path}")
                print("Please check the link again!")
                exit(1)
        
        print(f"TEST CONFIGURATION:")
        print(f"   - Level: {STAGE_NAME}")
        print(f"   - Model: {os.path.basename(model_path)}")
        print(f"   - Num steps: {NUM_EPISODES}")
        print(f"   - Record video: {'yes' if RECORD_VIDEO else 'No'}")
        print(f"   - Display: {'Yes' if RENDER_GAME else 'No'}")
        
        # input("Press Enter to start testing...") # Commenting out to avoid blocking unattended runs if any, although original had it. Adding it back for parity.
        input("Press Enter to start testing...")
        
        test_model(model_path, NUM_EPISODES, RENDER_GAME)
        
    except KeyboardInterrupt:
        print("Test interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise