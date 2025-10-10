"""
Custom Reward Wrapper cho Super Mario Bros
Tăng cường reward để Mario học tốt hơn
"""

import gym
import numpy as np


class CustomRewardWrapper(gym.Wrapper):
    """
    Custom reward wrapper kết hợp với reward gốc từ gym_super_mario_bros
    
    Reward gốc của gym_super_mario_bros:
    - Di chuyển sang phải: +1 mỗi pixel
    - Ăn coin: +50
    - Giết enemy: +100-500
    - Cắm cờ: +5000
    - Chết: -15
    
    Custom reward thêm vào:
    - Di chuyển nhanh sang phải: bonus
    - Chết: phạt nặng hơn
    - Đứng yên/di chuyển chậm: phạt nhẹ
    """
    
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        
        # ================================================================
        # CẤU HÌNH CUSTOM REWARD - TÙY CHỈNH TẠI ĐÂY
        # ================================================================
        
        # Reward cho việc di chuyển nhanh
        self.speed_reward_multiplier = 0.5  # Nhân với tốc độ di chuyển
        self.min_speed_threshold = 5        # Tốc độ tối thiểu để được thưởng
        
        # Penalty cho việc chết
        self.death_penalty = -50            # Phạt thêm khi chết (gốc là -15)
        
        # Penalty cho việc đứng yên
        self.idle_penalty = -0.1            # Phạt mỗi frame đứng yên
        self.idle_threshold = 2             # Nếu di chuyển < 2 pixel = đứng yên
        
        # Bonus cho việc tiến xa
        self.distance_bonus_multiplier = 1.5  # Nhân với khoảng cách di chuyển
        
        # ================================================================
        
        # Tracking variables
        self.prev_x_pos = 0
        self.prev_time = 400
        self.stuck_counter = 0
        self.max_x_pos = 0  # Vị trí xa nhất từng đạt được
        
    def reset(self, **kwargs):
        """Reset môi trường và các biến tracking"""
        obs = self.env.reset(**kwargs)
        self.prev_x_pos = 0
        self.prev_time = 400
        self.stuck_counter = 0
        self.max_x_pos = 0
        return obs
    
    def step(self, action):
        """
        Thực hiện action và tính custom reward
        """
        obs, reward, done, info = self.env.step(action)
        
        # Lấy thông tin từ game
        current_x_pos = info.get('x_pos', 0)
        current_time = info.get('time', 400)
        is_dead = info.get('life', 2) < 2  # Life giảm = chết
        
        # ============================================================
        # TÍNH CUSTOM REWARD
        # ============================================================
        custom_reward = 0
        
        # 1. REWARD CHO DI CHUYỂN NHANH SANG PHẢI
        x_delta = current_x_pos - self.prev_x_pos
        
        if x_delta > self.min_speed_threshold:
            # Di chuyển nhanh → thưởng
            speed_bonus = x_delta * self.speed_reward_multiplier
            custom_reward += speed_bonus
            self.stuck_counter = 0  # Reset counter
            
            # Bonus thêm nếu đạt vị trí xa nhất
            if current_x_pos > self.max_x_pos:
                distance_bonus = (current_x_pos - self.max_x_pos) * self.distance_bonus_multiplier
                custom_reward += distance_bonus
                self.max_x_pos = current_x_pos
                
        elif x_delta <= self.idle_threshold:
            # 2. PENALTY CHO ĐỨNG YÊN / DI CHUYỂN CHẬM
            custom_reward += self.idle_penalty
            self.stuck_counter += 1
            
            # Phạt nặng hơn nếu đứng yên quá lâu
            if self.stuck_counter > 50:  # Đứng yên > 50 frames
                custom_reward += self.idle_penalty * 2
            
            if self.stuck_counter > 100:  # Đứng yên > 100 frames
                custom_reward += self.idle_penalty * 5
        
        # 3. PENALTY NẶNG CHO CHẾT
        if is_dead:
            custom_reward += self.death_penalty
        
        # 4. PENALTY NHẸ CHO THỜI GIAN TRÔ
        # (Khuyến khích hoàn thành nhanh)
        time_penalty = -0.01 if current_time < self.prev_time else 0
        custom_reward += time_penalty
        
        # ============================================================
        # KẾT HỢP REWARD GỐC VÀ CUSTOM REWARD
        # ============================================================
        final_reward = reward + custom_reward
        
        # Update tracking variables
        self.prev_x_pos = current_x_pos
        self.prev_time = current_time
        
        # Debug info (optional - comment out nếu không cần)
        info['custom_reward'] = custom_reward
        info['original_reward'] = reward
        info['x_delta'] = x_delta
        
        return obs, final_reward, done, info


class DetailedRewardWrapper(gym.Wrapper):
    """
    Wrapper nâng cao với reward system chi tiết hơn
    Dành cho những ai muốn kiểm soát tốt hơn
    """
    
    def __init__(self, env, config=None):
        super(DetailedRewardWrapper, self).__init__(env)
        
        # Default configuration
        default_config = {
            # Movement rewards
            'right_movement_bonus': 1.0,      # Bonus khi di chuyển phải
            'fast_movement_bonus': 2.0,       # Bonus khi di chuyển rất nhanh
            'fast_threshold': 10,             # Ngưỡng để được coi là "nhanh"
            
            # Penalties
            'death_penalty': -100,            # Phạt khi chết
            'idle_penalty': -0.5,             # Phạt khi đứng yên
            'backward_penalty': -1.0,         # Phạt khi đi lùi
            'time_penalty': -0.01,            # Phạt theo thời gian
            
            # Progress rewards
            'new_area_bonus': 5.0,            # Bonus khi đến vùng mới
            'checkpoint_bonus': 10.0,         # Bonus tại các checkpoint
            
            # Thresholds
            'idle_threshold': 1,              # Ngưỡng coi là đứng yên
            'stuck_threshold': 50,            # Số frame để coi là "stuck"
        }
        
        # Merge với config người dùng
        self.config = {**default_config, **(config or {})}
        
        # Tracking
        self.prev_x_pos = 0
        self.max_x_pos = 0
        self.stuck_counter = 0
        self.prev_time = 400
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_x_pos = 0
        self.max_x_pos = 0
        self.stuck_counter = 0
        self.prev_time = 400
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        current_x_pos = info.get('x_pos', 0)
        current_time = info.get('time', 400)
        is_dead = info.get('life', 2) < 2
        
        custom_reward = 0
        
        # Movement analysis
        x_delta = current_x_pos - self.prev_x_pos
        
        # Right movement rewards
        if x_delta > 0:
            custom_reward += self.config['right_movement_bonus']
            self.stuck_counter = 0
            
            # Fast movement bonus
            if x_delta >= self.config['fast_threshold']:
                custom_reward += self.config['fast_movement_bonus']
            
            # New area bonus
            if current_x_pos > self.max_x_pos:
                progress = current_x_pos - self.max_x_pos
                custom_reward += progress * 0.1  # Small bonus for progress
                self.max_x_pos = current_x_pos
                
                # Checkpoint bonus (every 100 pixels)
                if self.max_x_pos % 100 < progress:
                    custom_reward += self.config['checkpoint_bonus']
        
        # Backward movement penalty
        elif x_delta < -self.config['idle_threshold']:
            custom_reward += self.config['backward_penalty']
        
        # Idle penalty
        elif abs(x_delta) <= self.config['idle_threshold']:
            custom_reward += self.config['idle_penalty']
            self.stuck_counter += 1
            
            # Escalating penalty for being stuck
            if self.stuck_counter > self.config['stuck_threshold']:
                stuck_multiplier = (self.stuck_counter - self.config['stuck_threshold']) / 10
                custom_reward += self.config['idle_penalty'] * stuck_multiplier
        
        # Death penalty
        if is_dead:
            custom_reward += self.config['death_penalty']
        
        # Time penalty (encourage fast completion)
        if current_time < self.prev_time:
            custom_reward += self.config['time_penalty']
        
        # Final reward
        final_reward = reward + custom_reward
        
        # Update tracking
        self.prev_x_pos = current_x_pos
        self.prev_time = current_time
        
        # Enhanced info
        info['custom_reward'] = custom_reward
        info['original_reward'] = reward
        info['x_delta'] = x_delta
        info['stuck_counter'] = self.stuck_counter
        info['max_x_pos'] = self.max_x_pos
        
        return obs, final_reward, done, info


# ============================================================================
# HELPER FUNCTION - Tạo môi trường với custom reward
# ============================================================================

def create_mario_env_with_custom_reward(world, stage, reward_type='custom', reward_config=None):
    """
    Tạo môi trường Mario với custom reward wrapper
    
    Args:
        world: World number (1-8)
        stage: Stage number (1-4)
        reward_type: 'custom' hoặc 'detailed'
        reward_config: Dict cấu hình cho DetailedRewardWrapper (optional)
    
    Returns:
        env: Môi trường đã được wrap
    """
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from gym.wrappers import GrayScaleObservation
    from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
    
    env_name = f'SuperMarioBros-{world}-{stage}-v0'
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    # Apply custom reward wrapper TRƯỚC khi grayscale
    if reward_type == 'custom':
        env = CustomRewardWrapper(env)
    elif reward_type == 'detailed':
        env = DetailedRewardWrapper(env, reward_config)
    
    # Tiếp tục với preprocessing
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    
    return env


# ============================================================================
# VÍ DỤ SỬ DỤNG
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CUSTOM REWARD WRAPPER - EXAMPLES")
    print("=" * 70)
    
    print("\n1. Sử dụng CustomRewardWrapper (Đơn giản):")
    print("-" * 70)
    print("""
    from custom_rewards import create_mario_env_with_custom_reward
    
    env = create_mario_env_with_custom_reward(
        world=1, 
        stage=1, 
        reward_type='custom'
    )
    """)
    
    print("\n2. Sử dụng DetailedRewardWrapper (Nâng cao):")
    print("-" * 70)
    print("""
    custom_config = {
        'right_movement_bonus': 2.0,
        'death_penalty': -150,
        'fast_threshold': 15,
    }
    
    env = create_mario_env_with_custom_reward(
        world=1, 
        stage=1, 
        reward_type='detailed',
        reward_config=custom_config
    )
    """)
    
    print("\n3. Tùy chỉnh trong CustomRewardWrapper:")
    print("-" * 70)
    print("""
    # Mở file custom_rewards.py và sửa các giá trị:
    
    self.speed_reward_multiplier = 1.0  # Tăng để thưởng nhiều hơn
    self.death_penalty = -100          # Tăng để phạt nặng hơn
    self.idle_penalty = -0.5           # Tăng để phạt đứng yên nhiều hơn
    """)