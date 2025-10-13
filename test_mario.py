
import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
from gym.wrappers import RecordVideo
import glob

# ============================================================================
# Select level to test - CHANGE HERE
# ============================================================================
WORLD = 1  # Must match trained world
STAGE = 1  # Must match trained stage


RECORD_VIDEO = True         # Is recording video?
NUM_EPISODES = 25           # Number of episodes to test
RENDER_GAME = True          # Render the game during testing

# choose model to test or None to select interactively
MODEL_PATH = None  # ex: "./train/1-1/best_model_100000" hoặc None


STAGE_NAME = f"{WORLD}-{STAGE}"
CHECKPOINT_DIR = f"./train/{STAGE_NAME}/"
VIDEO_DIR = f"./videos/{STAGE_NAME}/"

os.makedirs(VIDEO_DIR, exist_ok=True)

print("=" * 70)
print(f"🎮 KIỂM THỬ AI CHO SUPER MARIO BROS - MÀN {STAGE_NAME}")
print("=" * 70)

# ============================================================================
# CUSTOM REWARD MUST MATCH TRAIN
# ============================================================================
USE_CUSTOM_REWARD = True  # Use custom reward function?

if USE_CUSTOM_REWARD:
    try:
        from custom_rewards import CustomRewardWrapper
    except ImportError:
        print("⚠️  Không tìm thấy custom_rewards.py, sẽ dùng reward gốc")
        USE_CUSTOM_REWARD = False

# ============================================================================
# CREATE MARIO ENV
# ============================================================================
def create_mario_env(world, stage, record_video=False, video_folder=None):
    """
    Tạo môi trường Super Mario Bros cho việc kiểm thử
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
    
    model_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    
    if not model_files:
        raise FileNotFoundError(
            f"Not found model in {checkpoint_dir}\n"
            f"run train_mario.py before!"
        )
    
   
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model.replace('.zip', '')

def list_available_models(checkpoint_dir):
    
    model_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    
    if not model_files:
        return []
    
    models = []
    for f in sorted(model_files):
        name = os.path.basename(f).replace('.zip', '')
        size = os.path.getsize(f) / (1024 * 1024)  # MB
        models.append({
            'name': name,
            'path': f.replace('.zip', ''),
            'size_mb': round(size, 2)
        })
    
    return models

# ============================================================================
# TEST MODEL FUNCTION
# ============================================================================
def test_model(model_path, num_episodes=3, render=True):
    
    print(f"\n📦 Loading model: {model_path}")
    
    
    try:
        model = PPO.load(model_path)
        print("✅ Model load finish!")
    except Exception as e:
        print(f"❌ Error when load model: {e}")
        return
    
    
    video_folder = VIDEO_DIR if RECORD_VIDEO else None
    env = create_mario_env(WORLD, STAGE, RECORD_VIDEO, video_folder)
    
    if RECORD_VIDEO:
        print(f"🎬 Video will be save at: {VIDEO_DIR}")
    
    
    print(f"\n🎮 Started playing {num_episodes} time...")
    print("-" * 70)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n▶️  Episode {episode + 1}/{num_episodes}")
        
        while not done:
            # Dự đoán hành động
            action, _ = model.predict(state)
            
            # Thực hiện hành động
            state, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            
            # Render nếu cần
            if render:
                env.render()
        
        # Thông tin kết quả
        print(f"   ✓ Completed later {steps} steps")
        print(f"   🏆 Total reward: {total_reward:.2f}")
        
        if info[0].get('flag_get'):
            print(f"   🚩 FLAG IS UP! Mario has completed the level!")
        else:
            print(f"   💀 Mario is dead or out of time")
    
    print("\n" + "=" * 70)
    print("🎉 TEST COMPLETE!")
    
    if RECORD_VIDEO:
        print(f"🎬 The video has been saved at: {VIDEO_DIR}")
    
    print("=" * 70)
    
    env.close()

# ============================================================================
# INTERACTIVE MODEL SELECTION
# ============================================================================
def interactive_model_selection():
    
    models = list_available_models(CHECKPOINT_DIR)
    
    if not models:
        print(f"\n❌ No model found in {CHECKPOINT_DIR}")
        print("   Run train_mario.py first!")
        return None
    
    print("\n📋 AVAILABLE MODELS:")
    print("-" * 70)
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']} ({model['size_mb']} MB)")
    print("-" * 70)
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}) or Enter to select latest: ").strip()
            
            if choice == "":
                
                return models[-1]['path']
            
            choice = int(choice)
            if 1 <= choice <= len(models):
                return models[choice - 1]['path']
            else:
                print(f"❌ Please select a number from 1 to {len(models)}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n⚠️  Cancelled")
            return None

# ============================================================================
# RUN PROGRAM
# ============================================================================
if __name__ == "__main__":
    try:
        if MODEL_PATH is None:
            print("\n🔍 Looking for model...")
            model_path = interactive_model_selection()
            
            if model_path is None:
                print("\n❌ No model to test. Exit...")
                exit(1)
        else:
            model_path = MODEL_PATH
            
            if not os.path.exists(model_path + '.zip'):
                print(f"\n❌ Model does not exist: {model_path}")
                print("   Please check the link again!")
                exit(1)
        
        # Hiển thị cấu hình
        print(f"\n⚙️  TEST CONFIGURATION:")
        print(f"   - Level: {STAGE_NAME}")
        print(f"   - Model: {os.path.basename(model_path)}")
        print(f"   - Num steps: {NUM_EPISODES}")
        print(f"   - Record video: {'yes' if RECORD_VIDEO else 'No'}")
        print(f"   - Display: {'Yes' if RENDER_GAME else 'No'}")
        
        # Xác nhận
        input("\n▶️  Press Enter to start testing...")
        
        # Chạy test
        test_model(model_path, NUM_EPISODES, RENDER_GAME)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise