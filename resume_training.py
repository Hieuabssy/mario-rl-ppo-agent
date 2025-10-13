"""
Resume Training Script - Tiếp tục huấn luyện từ checkpoint
"""

import os
import glob
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# ============================================================================
# CẤU HÌNH - THAY ĐỔI TẠI ĐÂY
# ============================================================================
WORLD = 1
STAGE = 1

# Cấu hình resume
CHECKPOINT_TIMESTEP = 2000000       # Timestep của checkpoint muốn load
ADDITIONAL_TIMESTEPS = 500000       # Số timestep muốn train thêm
CHECKPOINT_FREQ = 100000             # Lưu checkpoint mỗi bao nhiêu bước

# Cấu hình custom reward
USE_CUSTOM_REWARD = False

if USE_CUSTOM_REWARD:
    try:
        from custom_rewards import CustomRewardWrapper
        print("✅ Custom reward wrapper đã được import!")
    except ImportError:
        print("⚠️  Không tìm thấy custom_rewards.py, sẽ dùng reward gốc")
        USE_CUSTOM_REWARD = False

# ============================================================================
# THIẾT LẬP THƯ MỤC
# ============================================================================
STAGE_NAME = f"{WORLD}-{STAGE}"
CHECKPOINT_DIR = f"./train/{STAGE_NAME}/"
LOG_DIR = f"./logs/{STAGE_NAME}/"

print("=" * 70)
print(f"🔄 RESUME TRAINING - MÀN {STAGE_NAME}")
print("=" * 70)
print(f"📦 Checkpoint: best_model_{CHECKPOINT_TIMESTEP}")
print(f"➕ Train thêm: {ADDITIONAL_TIMESTEPS:,} timesteps")
print(f"📁 Thư mục: {CHECKPOINT_DIR}")
print("=" * 70)

# ============================================================================
# CALLBACK
# ============================================================================
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, start_timestep=0, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.start_timestep = start_timestep  # Timestep bắt đầu

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        # Tính tổng timestep thực tế (bao gồm cả checkpoint cũ)
        total_timesteps = self.start_timestep + self.n_calls
        
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(
                self.save_path, 
                f'best_model_{total_timesteps}'
            )
            self.model.save(model_path)
            print(f"\n💾 Đã lưu checkpoint tại bước {total_timesteps:,}")
        return True

# ============================================================================
# TẠO MÔI TRƯỜNG
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
    
    print("✅ Môi trường đã được tạo thành công!")
    return env

# ============================================================================
# TÌM VÀ LOAD CHECKPOINT
# ============================================================================
def find_checkpoint(checkpoint_dir, timestep):
    """
    Tìm checkpoint với timestep cụ thể
    """
    checkpoint_path = os.path.join(checkpoint_dir, f'best_model_{timestep}')
    
    if os.path.exists(checkpoint_path + '.zip'):
        return checkpoint_path
    
    # Nếu không tìm thấy, list tất cả checkpoints có sẵn
    print(f"\n❌ Không tìm thấy checkpoint: {checkpoint_path}")
    print("\n📋 CÁC CHECKPOINT CÓ SẴN:")
    print("-" * 70)
    
    model_files = glob.glob(os.path.join(checkpoint_dir, "best_model_*.zip"))
    
    if not model_files:
        print("   Không có checkpoint nào!")
        return None
    
    checkpoints = []
    for f in sorted(model_files):
        name = os.path.basename(f).replace('.zip', '')
        # Extract timestep from name
        try:
            ts = int(name.split('_')[-1])
            checkpoints.append({'name': name, 'path': f.replace('.zip', ''), 'timestep': ts})
        except:
            pass
    
    for i, cp in enumerate(checkpoints, 1):
        print(f"{i}. {cp['name']} ({cp['timestep']:,} steps)")
    
    print("-" * 70)
    
    # Cho phép người dùng chọn
    while True:
        try:
            choice = input(f"\nChọn checkpoint (1-{len(checkpoints)}) hoặc 'q' để thoát: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            choice = int(choice)
            if 1 <= choice <= len(checkpoints):
                selected = checkpoints[choice - 1]
                return selected['path'], selected['timestep']
            else:
                print(f"❌ Vui lòng chọn số từ 1 đến {len(checkpoints)}")
        except ValueError:
            print("❌ Vui lòng nhập số hợp lệ hoặc 'q'")
        except KeyboardInterrupt:
            print("\n\n⚠️  Đã hủy")
            return None

# ============================================================================
# RESUME TRAINING
# ============================================================================
def resume_training():
    """
    Load checkpoint và tiếp tục training
    """
    # Tìm checkpoint
    print("\n🔍 Đang tìm checkpoint...")
    
    result = find_checkpoint(CHECKPOINT_DIR, CHECKPOINT_TIMESTEP)
    
    if result is None:
        print("\n❌ Không thể tiếp tục training. Thoát...")
        return
    
    if isinstance(result, tuple):
        checkpoint_path, actual_timestep = result
    else:
        checkpoint_path = result
        actual_timestep = CHECKPOINT_TIMESTEP
    
    print(f"\n✅ Tìm thấy checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"📊 Timestep hiện tại: {actual_timestep:,}")
    
    # Tạo môi trường
    env = create_mario_env(WORLD, STAGE)
    
    # Load model từ checkpoint
    print(f"\n📦 Đang load model từ checkpoint...")
    try:
        model = PPO.load(
            checkpoint_path,
            env=env,
            tensorboard_log=LOG_DIR
        )
        print("✅ Model đã được load thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        return
    
    # Hiển thị thông tin
    print("\n" + "=" * 70)
    print("📊 THÔNG TIN TRAINING")
    print("=" * 70)
    print(f"🎮 Màn chơi: {STAGE_NAME}")
    print(f"📍 Bắt đầu từ: {actual_timestep:,} steps")
    print(f"➕ Train thêm: {ADDITIONAL_TIMESTEPS:,} steps")
    print(f"🎯 Tổng sau khi train: {actual_timestep + ADDITIONAL_TIMESTEPS:,} steps")
    print(f"💾 Checkpoint mỗi: {CHECKPOINT_FREQ:,} steps")
    print("=" * 70)
    
    # Xác nhận
    confirm = input("\n▶️  Tiếp tục? (y/n): ").strip().lower()
    if confirm != 'y':
        print("⚠️  Đã hủy")
        return
    
    # Tạo callback với offset timestep
    callback = TrainAndLoggingCallback(
        check_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        start_timestep=actual_timestep  # Bắt đầu từ timestep cũ
    )
    
    # Tiếp tục training
    print("\n🚀 BẮT ĐẦU TIẾP TỤC TRAINING...")
    print("-" * 70)
    
    try:
        model.learn(
            total_timesteps=ADDITIONAL_TIMESTEPS,
            callback=callback,
            reset_num_timesteps=False  # QUAN TRỌNG: Không reset timesteps
        )
        
        # Lưu final model
        final_timestep = actual_timestep + ADDITIONAL_TIMESTEPS
        final_model_path = os.path.join(
            CHECKPOINT_DIR, 
            f'best_model_{final_timestep}'
        )
        model.save(final_model_path)
        
        print("\n" + "=" * 70)
        print("🎉 HOÀN THÀNH!")
        print("=" * 70)
        print(f"💾 Final model: {final_model_path}")
        print(f"📊 Tổng timesteps: {final_timestep:,}")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training bị gián đoạn!")
        print("💾 Model đã được lưu tại các checkpoint trước đó")
    
    env.close()

# ============================================================================
# UTILITY: List tất cả checkpoints
# ============================================================================
def list_all_checkpoints():
    """
    Hiển thị tất cả checkpoints có sẵn
    """
    print("\n" + "=" * 70)
    print(f"📋 TẤT CẢ CHECKPOINTS - MÀN {STAGE_NAME}")
    print("=" * 70)
    
    model_files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.zip"))
    
    if not model_files:
        print("❌ Không có checkpoint nào!")
        return
    
    checkpoints = []
    for f in sorted(model_files):
        name = os.path.basename(f).replace('.zip', '')
        size = os.path.getsize(f) / (1024 * 1024)  # MB
        mtime = os.path.getmtime(f)
        
        # Extract timestep
        try:
            if 'best_model_' in name:
                ts = int(name.split('_')[-1])
            else:
                ts = 0
        except:
            ts = 0
        
        checkpoints.append({
            'name': name,
            'timestep': ts,
            'size_mb': round(size, 2),
            'modified': mtime
        })
    
    # Sort by timestep
    checkpoints.sort(key=lambda x: x['timestep'])
    
    print(f"\n{'#':<4} {'Tên':<30} {'Timesteps':<15} {'Size (MB)':<12}")
    print("-" * 70)
    
    for i, cp in enumerate(checkpoints, 1):
        ts_str = f"{cp['timestep']:,}" if cp['timestep'] > 0 else "N/A"
        print(f"{i:<4} {cp['name']:<30} {ts_str:<15} {cp['size_mb']:<12}")
    
    print("-" * 70)
    print(f"Tổng: {len(checkpoints)} checkpoints")
    print("=" * 70)

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import sys
    
    # Check arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        list_all_checkpoints()
    else:
        try:
            resume_training()
        except KeyboardInterrupt:
            print("\n\n⚠️  Đã hủy bởi người dùng")
        except Exception as e:
            print(f"\n\n❌ Lỗi: {e}")
            raise