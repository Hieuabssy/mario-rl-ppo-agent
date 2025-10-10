"""
Super Mario Bros AI Testing Script
Kiểm thử các model đã huấn luyện với tùy chọn quay video
"""

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
# CẤU HÌNH KIỂM THỬ - THAY ĐỔI TẠI ĐÂY
# ============================================================================
WORLD = 1  # Phải khớp với màn đã train
STAGE = 1  # Phải khớp với màn đã train

# Tùy chọn kiểm thử
RECORD_VIDEO = True         # Có quay video không?
NUM_EPISODES = 3            # Số lượt chơi
RENDER_GAME = True          # Hiển thị game trong khi chơi

# Chọn model để test (có thể thay đổi)
# Để trống để tự động chọn model mới nhất, hoặc chỉ định đường dẫn cụ thể
MODEL_PATH = None  # Ví dụ: "./train/1-1/best_model_100000" hoặc None

# ============================================================================
# THIẾT LẬP THƯ MỤC
# ============================================================================
STAGE_NAME = f"{WORLD}-{STAGE}"
CHECKPOINT_DIR = f"./train/{STAGE_NAME}/"
VIDEO_DIR = f"./videos/{STAGE_NAME}/"

os.makedirs(VIDEO_DIR, exist_ok=True)

print("=" * 70)
print(f"🎮 KIỂM THỬ AI CHO SUPER MARIO BROS - MÀN {STAGE_NAME}")
print("=" * 70)

# ============================================================================
# CẤU HÌNH CUSTOM REWARD (PHẢI KHỚP VỚI TRAIN)
# ============================================================================
USE_CUSTOM_REWARD = True  # Đặt False nếu train không dùng custom reward

if USE_CUSTOM_REWARD:
    try:
        from custom_rewards import CustomRewardWrapper
    except ImportError:
        print("⚠️  Không tìm thấy custom_rewards.py, sẽ dùng reward gốc")
        USE_CUSTOM_REWARD = False

# ============================================================================
# HÀM TẠO MÔI TRƯỜNG
# ============================================================================
def create_mario_env(world, stage, record_video=False, video_folder=None):
    """
    Tạo môi trường Super Mario Bros cho việc kiểm thử
    """
    env_name = f'SuperMarioBros-{world}-{stage}-v0'
    env = gym_super_mario_bros.make(env_name)
    
    # Nếu cần quay video, wrap trước khi apply các wrapper khác
    if record_video and video_folder:
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda x: True,  # Quay tất cả các episode
            name_prefix=f"mario_{STAGE_NAME}_test"
        )
    
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    # Apply custom reward nếu cần (phải khớp với training)
    if USE_CUSTOM_REWARD:
        env = CustomRewardWrapper(env)
    
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    
    return env

# ============================================================================
# HÀM TÌM MODEL
# ============================================================================
def find_latest_model(checkpoint_dir):
    """
    Tìm model mới nhất trong thư mục checkpoint
    """
    # Tìm tất cả các file model
    model_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    
    if not model_files:
        raise FileNotFoundError(
            f"Không tìm thấy model nào trong {checkpoint_dir}\n"
            f"Hãy chạy train_mario.py trước!"
        )
    
    # Sắp xếp theo thời gian sửa đổi
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model.replace('.zip', '')

def list_available_models(checkpoint_dir):
    """
    Liệt kê tất cả các model có sẵn
    """
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
# HÀM KIỂM THỬ
# ============================================================================
def test_model(model_path, num_episodes=3, render=True):
    """
    Kiểm thử model đã huấn luyện
    """
    print(f"\n📦 Đang tải model: {model_path}")
    
    # Load model
    try:
        model = PPO.load(model_path)
        print("✅ Model đã được tải thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi tải model: {e}")
        return
    
    # Tạo môi trường
    video_folder = VIDEO_DIR if RECORD_VIDEO else None
    env = create_mario_env(WORLD, STAGE, RECORD_VIDEO, video_folder)
    
    if RECORD_VIDEO:
        print(f"🎬 Video sẽ được lưu tại: {VIDEO_DIR}")
    
    # Chạy các episode
    print(f"\n🎮 Bắt đầu chơi {num_episodes} lượt...")
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
        print(f"   ✓ Hoàn thành sau {steps} bước")
        print(f"   🏆 Tổng điểm: {total_reward:.2f}")
        
        if info[0].get('flag_get'):
            print(f"   🚩 ĐÃ CẮM CỜ! Mario đã hoàn thành màn!")
        else:
            print(f"   💀 Mario đã chết hoặc hết thời gian")
    
    print("\n" + "=" * 70)
    print("🎉 HOÀN THÀNH KIỂM THỬ!")
    
    if RECORD_VIDEO:
        print(f"🎬 Video đã được lưu tại: {VIDEO_DIR}")
    
    print("=" * 70)
    
    env.close()

# ============================================================================
# HÀM CHỌN MODEL TƯƠNG TÁC
# ============================================================================
def interactive_model_selection():
    """
    Cho phép người dùng chọn model từ danh sách
    """
    models = list_available_models(CHECKPOINT_DIR)
    
    if not models:
        print(f"\n❌ Không tìm thấy model nào trong {CHECKPOINT_DIR}")
        print("   Hãy chạy train_mario.py trước!")
        return None
    
    print("\n📋 CÁC MODEL CÓ SẴN:")
    print("-" * 70)
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']} ({model['size_mb']} MB)")
    print("-" * 70)
    
    while True:
        try:
            choice = input(f"\nChọn model (1-{len(models)}) hoặc Enter để chọn mới nhất: ").strip()
            
            if choice == "":
                # Chọn model mới nhất (cuối cùng trong list)
                return models[-1]['path']
            
            choice = int(choice)
            if 1 <= choice <= len(models):
                return models[choice - 1]['path']
            else:
                print(f"❌ Vui lòng chọn số từ 1 đến {len(models)}")
        except ValueError:
            print("❌ Vui lòng nhập số hợp lệ")
        except KeyboardInterrupt:
            print("\n\n⚠️  Đã hủy")
            return None

# ============================================================================
# CHẠY CHƯƠNG TRÌNH
# ============================================================================
if __name__ == "__main__":
    try:
        # Tìm hoặc chọn model
        if MODEL_PATH is None:
            print("\n🔍 Đang tìm model...")
            model_path = interactive_model_selection()
            
            if model_path is None:
                print("\n❌ Không có model để test. Thoát...")
                exit(1)
        else:
            model_path = MODEL_PATH
            
            if not os.path.exists(model_path + '.zip'):
                print(f"\n❌ Model không tồn tại: {model_path}")
                print("   Vui lòng kiểm tra lại đường dẫn!")
                exit(1)
        
        # Hiển thị cấu hình
        print(f"\n⚙️  CẤU HÌNH KIỂM THỬ:")
        print(f"   - Màn chơi: {STAGE_NAME}")
        print(f"   - Model: {os.path.basename(model_path)}")
        print(f"   - Số lượt: {NUM_EPISODES}")
        print(f"   - Quay video: {'Có' if RECORD_VIDEO else 'Không'}")
        print(f"   - Hiển thị: {'Có' if RENDER_GAME else 'Không'}")
        
        # Xác nhận
        input("\n▶️  Nhấn Enter để bắt đầu kiểm thử...")
        
        # Chạy test
        test_model(model_path, NUM_EPISODES, RENDER_GAME)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Kiểm thử bị gián đoạn bởi người dùng")
    except Exception as e:
        print(f"\n\n❌ Lỗi: {e}")
        raise