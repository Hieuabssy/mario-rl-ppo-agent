"""
Super Mario Bros AI Training Script
Huấn luyện mô hình AI chơi Super Mario Bros với khả năng tùy chỉnh màn chơi
"""

import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# ============================================================================
# CẤU HÌNH MÀN CHƠI - THAY ĐỔI TẠI ĐÂY
# ============================================================================
WORLD = 1  # Chọn world (1-8)
STAGE = 1  # Chọn stage (1-4)

# ============================================================================
# CẤU HÌNH HUẤN LUYỆN
# ============================================================================
TOTAL_TIMESTEPS = 100000  # Tổng số bước huấn luyện
CHECKPOINT_FREQ = 10000    # Lưu checkpoint mỗi bao nhiêu bước
LEARNING_RATE = 0.000001   # Tốc độ học
N_STEPS = 512              # Số bước mỗi update

# ============================================================================
# TẠO CẤU TRÚC THƯ MỤC THEO MÀN CHƠI
# ============================================================================
STAGE_NAME = f"{WORLD}-{STAGE}"
CHECKPOINT_DIR = f"./train/{STAGE_NAME}/"
LOG_DIR = f"./logs/{STAGE_NAME}/"
VIDEO_DIR = f"./videos/{STAGE_NAME}/"

# Tạo các thư mục nếu chưa tồn tại
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

print("=" * 70)
print(f"🎮 HUẤN LUYỆN AI CHO SUPER MARIO BROS - MÀN {STAGE_NAME}")
print("=" * 70)
print(f"📁 Checkpoint: {CHECKPOINT_DIR}")
print(f"📊 Logs: {LOG_DIR}")
print(f"🎬 Videos: {VIDEO_DIR}")
print("=" * 70)

# ============================================================================
# CALLBACK ĐỂ LƯU MODEL ĐỊNH KỲ
# ============================================================================
class TrainAndLoggingCallback(BaseCallback):
    """
    Callback tùy chỉnh để lưu model định kỳ trong quá trình huấn luyện
    """
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
# CẤU HÌNH CUSTOM REWARD (TÙY CHỌN)
# ============================================================================
USE_CUSTOM_REWARD = True  # Đặt False để dùng reward gốc

# Nếu muốn tùy chỉnh chi tiết, import custom reward wrapper
if USE_CUSTOM_REWARD:
    try:
        from custom_rewards import CustomRewardWrapper
        print("✅ Custom reward wrapper đã được import!")
    except ImportError:
        print("⚠️  Không tìm thấy custom_rewards.py, sẽ dùng reward gốc")
        USE_CUSTOM_REWARD = False

# ============================================================================
# TẠO MÔI TRƯỜNG GAME
# ============================================================================
def create_mario_env(world, stage):
    """
    Tạo môi trường Super Mario Bros với các wrapper cần thiết
    
    Args:
        world: Số world (1-8)
        stage: Số stage (1-4)
    
    Returns:
        env: Môi trường đã được xử lý sẵn
    """
    # Tạo tên môi trường động dựa trên world và stage
    env_name = f'SuperMarioBros-{world}-{stage}-v0'
    print(f"\n🌍 Đang tạo môi trường: {env_name}")
    
    # 1. Tạo môi trường cơ bản
    env = gym_super_mario_bros.make(env_name)
    
    # 2. Đơn giản hóa điều khiển
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    # 2.5. Apply Custom Reward Wrapper (TRƯỚC grayscale)
    if USE_CUSTOM_REWARD:
        env = CustomRewardWrapper(env)
        print("🎯 Custom reward wrapper đã được áp dụng!")
    
    # 3. Chuyển sang grayscale để giảm dữ liệu đầu vào
    env = GrayScaleObservation(env, keep_dim=True)
    
    # 4. Wrap trong Dummy Environment để vectorize
    env = DummyVecEnv([lambda: env])
    
    # 5. Stack 4 frames liên tiếp để AI có thể "nhớ" chuyển động
    env = VecFrameStack(env, 4, channels_order='last')
    
    print("✅ Môi trường đã được tạo và tiền xử lý thành công!")
    return env

# ============================================================================
# HUẤN LUYỆN MODEL
# ============================================================================
def train_model():
    """
    Huấn luyện mô hình PPO cho Super Mario Bros
    """
    print("\n" + "=" * 70)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN")
    print("=" * 70)
    
    # Tạo môi trường
    env = create_mario_env(WORLD, STAGE)
    
    # Tạo callback để lưu model
    callback = TrainAndLoggingCallback(
        check_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR
    )
    
    # Khởi tạo model PPO với CNN policy
    print("\n🤖 Đang khởi tạo mô hình PPO...")
    model = PPO(
        'CnnPolicy',           # Sử dụng CNN để xử lý hình ảnh
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS
    )
    print("✅ Mô hình đã được khởi tạo!")
    
    # Bắt đầu huấn luyện
    print(f"\n🎓 Bắt đầu huấn luyện với {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"⏱️  Checkpoint sẽ được lưu mỗi {CHECKPOINT_FREQ:,} bước")
    print("\n" + "-" * 70)
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback
    )
    
    # Lưu model cuối cùng
    final_model_path = os.path.join(CHECKPOINT_DIR, f'final_model_{STAGE_NAME}')
    model.save(final_model_path)
    
    print("\n" + "=" * 70)
    print("🎉 HOÀN THÀNH HUẤN LUYỆN!")
    print(f"💾 Model cuối cùng đã được lưu tại: {final_model_path}")
    print("=" * 70)
    
    env.close()
    return final_model_path

# ============================================================================
# CHẠY CHƯƠNG TRÌNH
# ============================================================================
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