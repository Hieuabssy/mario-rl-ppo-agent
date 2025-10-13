import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# ============================================================================
# Select level to train - CHANGE HERE
# ============================================================================
WORLD = 1  # Select world (1-8)
STAGE = 1  # Select stage (1-4)


TOTAL_TIMESTEPS = 4000000  # total training steps
CHECKPOINT_FREQ = 100000    # Save checkpoint every 100,000 steps
LEARNING_RATE = 0.000001   # Learning rate
N_STEPS = 512              # Number of steps per update


STAGE_NAME = f"{WORLD}-{STAGE}"
CHECKPOINT_DIR = f"./train/{STAGE_NAME}/"
LOG_DIR = f"./logs/{STAGE_NAME}/"
VIDEO_DIR = f"./videos/{STAGE_NAME}/"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

print("=" * 70)
print(f"🎮 TRAINING AI FOR SUPER MARIO BROS - LEVEL {STAGE_NAME}")
print("=" * 70)
print(f"📁 Checkpoint: {CHECKPOINT_DIR}")
print(f"📊 Logs: {LOG_DIR}")
print(f"🎬 Videos: {VIDEO_DIR}")
print("=" * 70)

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
                f'best_model_{self.n_calls}'
            )
            self.model.save(model_path)
            print(f"\n💾 Đã lưu checkpoint tại bước {self.n_calls}")
        return True

# ============================================================================
# CUSTOM REWARD CONFIGURATION (MUST MATCH TRAIN)
# ============================================================================
USE_CUSTOM_REWARD = False  #    Set to False if not using custom reward

# Apply custom reward wrapper if enabled
if USE_CUSTOM_REWARD:
    try:
        from custom_rewards import CustomRewardWrapper
        print("✅ Custom reward wrapper đã được import!")
    except ImportError:
        print("⚠️  Không tìm thấy custom_rewards.py, sẽ dùng reward gốc")
        USE_CUSTOM_REWARD = False

# ============================================================================
# ENV CREATION FUNCTION
# ============================================================================
def create_mario_env(world, stage):
  
   
    env_name = f'SuperMarioBros-{world}-{stage}-v0'
    print(f"\n🌍 Đang tạo môi trường: {env_name}")
    
    
    env = gym_super_mario_bros.make(env_name)
    
    
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
   
    if USE_CUSTOM_REWARD:
        env = CustomRewardWrapper(env)
        print("🎯 Custom reward wrapper đã được áp dụng!")
    
   
    env = GrayScaleObservation(env, keep_dim=True)
    
   
    env = DummyVecEnv([lambda: env])
    
    
    env = VecFrameStack(env, 4, channels_order='last')
    
    print("✅ Môi trường đã được tạo và tiền xử lý thành công!")
    return env

# ============================================================================
# Train model function
# ============================================================================
def train_model():
    """
    Huấn luyện mô hình PPO cho Super Mario Bros
    """
    print("\n" + "=" * 70)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN")
    print("=" * 70)
    
    
    env = create_mario_env(WORLD, STAGE)
    
    
    callback = TrainAndLoggingCallback(
        check_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR
    )
    
    
    print("\n🤖 Đang khởi tạo mô hình PPO...")
    model = PPO(
        'CnnPolicy',           
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS
    )
    print("✅ Mô hình đã được khởi tạo!")
    
    
    print(f"\n🎓 Bắt đầu huấn luyện với {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"⏱️  Checkpoint sẽ được lưu mỗi {CHECKPOINT_FREQ:,} bước")
    print("\n" + "-" * 70)
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback
    )
    
    
    final_model_path = os.path.join(CHECKPOINT_DIR, f'final_model_{STAGE_NAME}')
    model.save(final_model_path)
    
    print("\n" + "=" * 70)
    print("🎉 HOÀN THÀNH HUẤN LUYỆN!")
    print(f"💾 Model cuối cùng đã được lưu tại: {final_model_path}")
    print("=" * 70)
    
    env.close()
    return final_model_path


if __name__ == "__main__":
    try:
        final_model = train_model()
        print(f"\n✨ Sử dụng 'test_mario.py' để kiểm thử model!")
        print(f"   Model path: {final_model}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Huấn luyện bị gián đoạn bởi người dùng")
    except Exception as e:
        print(f"\n\n❌ Lỗi: {e}")
        raise