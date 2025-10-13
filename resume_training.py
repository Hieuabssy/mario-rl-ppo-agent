import os
import glob
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


WORLD = 1
STAGE = 1


CHECKPOINT_TIMESTEP = 2000000       # Timestep of the checkpoint to load
ADDITIONAL_TIMESTEPS = 500000       # Number of additional timesteps to train
CHECKPOINT_FREQ = 100000             # Save checkpoint every N steps


USE_CUSTOM_REWARD = False

if USE_CUSTOM_REWARD:
    try:
        from custom_rewards import CustomRewardWrapper
        print("✅ Custom reward wrapper imported successfully!")
    except ImportError:
        print("⚠️  custom_rewards.py not found, using default reward")
        USE_CUSTOM_REWARD = False


STAGE_NAME = f"{WORLD}-{STAGE}"
CHECKPOINT_DIR = f"./train/{STAGE_NAME}/"
LOG_DIR = f"./logs/{STAGE_NAME}/"

print("=" * 70)
print(f"🔄 RESUME TRAINING - STAGE {STAGE_NAME}")
print("=" * 70)
print(f"📦 Checkpoint: best_model_{CHECKPOINT_TIMESTEP}")
print(f"➕ Additional training: {ADDITIONAL_TIMESTEPS:,} timesteps")
print(f"📁 Directory: {CHECKPOINT_DIR}")
print("=" * 70)

# ============================================================================
# CALLBACK
# ============================================================================
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, start_timestep=0, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.start_timestep = start_timestep  

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        
        total_timesteps = self.start_timestep + self.n_calls
        
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(
                self.save_path, 
                f'best_model_{total_timesteps}'
            )
            self.model.save(model_path)
            print(f"\n💾 Checkpoint saved at step {total_timesteps:,}")
        return True

# ============================================================================
# CREATE ENVIRONMENT
# ============================================================================
def create_mario_env(world, stage):
    env_name = f'SuperMarioBros-{world}-{stage}-v0'
    print(f"\n🌍 Creating environment: {env_name}")
    
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    if USE_CUSTOM_REWARD:
        env = CustomRewardWrapper(env)
        print("🎯 Custom reward wrapper has been applied!")
    
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    
    print("✅ Environment created successfully!")
    return env

# ============================================================================
# FIND AND LOAD CHECKPOINT
# ============================================================================
def find_checkpoint(checkpoint_dir, timestep):
 
    checkpoint_path = os.path.join(checkpoint_dir, f'best_model_{timestep}')
    
    if os.path.exists(checkpoint_path + '.zip'):
        return checkpoint_path
    
    print(f"\n❌ Checkpoint not found: {checkpoint_path}")
    print("\n📋 AVAILABLE CHECKPOINTS:")
    print("-" * 70)
    
    model_files = glob.glob(os.path.join(checkpoint_dir, "best_model_*.zip"))
    
    if not model_files:
        print("   No checkpoints available!")
        return None
    
    checkpoints = []
    for f in sorted(model_files):
        name = os.path.basename(f).replace('.zip', '')
        try:
            ts = int(name.split('_')[-1])
            checkpoints.append({'name': name, 'path': f.replace('.zip', ''), 'timestep': ts})
        except:
            pass
    
    for i, cp in enumerate(checkpoints, 1):
        print(f"{i}. {cp['name']} ({cp['timestep']:,} steps)")
    
    print("-" * 70)
    while True:
        try:
            choice = input(f"\nSelect a checkpoint (1-{len(checkpoints)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            choice = int(choice)
            if 1 <= choice <= len(checkpoints):
                selected = checkpoints[choice - 1]
                return selected['path'], selected['timestep']
            else:
                print(f"❌ Please select a number from 1 to {len(checkpoints)}")
        except ValueError:
            print("❌ Please enter a valid number or 'q'")
        except KeyboardInterrupt:
            print("\n\n⚠️  Cancelled")
            return None

# ============================================================================
# RESUME TRAINING
# ============================================================================
def resume_training():

    print("\n🔍 Searching for checkpoint...")
    
    result = find_checkpoint(CHECKPOINT_DIR, CHECKPOINT_TIMESTEP)
    
    if result is None:
        print("\n❌ Cannot continue training. Exiting...")
        return
    
    if isinstance(result, tuple):
        checkpoint_path, actual_timestep = result
    else:
        checkpoint_path = result
        actual_timestep = CHECKPOINT_TIMESTEP
    
    print(f"\n✅ Checkpoint found: {os.path.basename(checkpoint_path)}")
    print(f"📊 Current timestep: {actual_timestep:,}")

    env = create_mario_env(WORLD, STAGE)
    
    print(f"\n📦 Loading model from checkpoint...")
    try:
        model = PPO.load(
            checkpoint_path,
            env=env,
            tensorboard_log=LOG_DIR
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("\n" + "=" * 70)
    print("📊 TRAINING INFORMATION")
    print("=" * 70)
    print(f"🎮 Stage: {STAGE_NAME}")
    print(f"📍 Starting from: {actual_timestep:,} steps")
    print(f"➕ Additional training: {ADDITIONAL_TIMESTEPS:,} steps")
    print(f"🎯 Total after training: {actual_timestep + ADDITIONAL_TIMESTEPS:,} steps")
    print(f"💾 Checkpoint every: {CHECKPOINT_FREQ:,} steps")
    print("=" * 70)
    
    confirm = input("\n▶️  Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("⚠️  Cancelled")
        return
    
    callback = TrainAndLoggingCallback(
        check_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        start_timestep=actual_timestep  # Start from the old timestep
    )
    
    print("\n🚀 STARTING RESUMED TRAINING...")
    print("-" * 70)
    
    try:
        model.learn(
            total_timesteps=ADDITIONAL_TIMESTEPS,
            callback=callback,
            reset_num_timesteps=False  
        )
        
        final_timestep = actual_timestep + ADDITIONAL_TIMESTEPS
        final_model_path = os.path.join(
            CHECKPOINT_DIR, 
            f'best_model_{final_timestep}'
        )
        model.save(final_model_path)
        
        print("\n" + "=" * 70)
        print("🎉 COMPLETE!")
        print("=" * 70)
        print(f"💾 Final model: {final_model_path}")
        print(f"📊 Total timesteps: {final_timestep:,}")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted!")
        print("💾 Model has been saved at previous checkpoints")
    
    env.close()

# ============================================================================
# UTILITY: List all checkpoints
# ============================================================================
def list_all_checkpoints():
    print("\n" + "=" * 70)
    print(f"📋 ALL CHECKPOINTS - STAGE {STAGE_NAME}")
    print("=" * 70)
    
    model_files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.zip"))
    
    if not model_files:
        print("❌ No checkpoints found!")
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
    
    checkpoints.sort(key=lambda x: x['timestep'])
    
    print(f"\n{'#':<4} {'Name':<30} {'Timesteps':<15} {'Size (MB)':<12}")
    print("-" * 70)
    
    for i, cp in enumerate(checkpoints, 1):
        ts_str = f"{cp['timestep']:,}" if cp['timestep'] > 0 else "N/A"
        print(f"{i:<4} {cp['name']:<30} {ts_str:<15} {cp['size_mb']:<12}")
    
    print("-" * 70)
    print(f"Total: {len(checkpoints)} checkpoints")
    print("=" * 70)

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        list_all_checkpoints()
    else:
        try:
            resume_training()
        except KeyboardInterrupt:
            print("\n\n⚠️  Cancelled by user")
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            raise