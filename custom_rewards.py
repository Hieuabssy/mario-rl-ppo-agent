
import gym
import numpy as np


class CustomRewardWrapper(gym.Wrapper):
    """
    Custom reward wrapper that combines with the original reward from gym_super_mario_bros.
    
    Original rewards from gym_super_mario_bros:
    - Moving to the right: +1 per pixel
    - Collecting a coin: +50
    - Killing an enemy: +100-500
    - Reaching the flagpole: +5000
    - Dying: -15
    
    Additional custom rewards:
    - Bonus for moving quickly to the right
    - Heavier penalty for dying
    - Slight penalty for standing still or moving slowly
    """
    # ================================================================
    # CUSTOM REWARD CONFIGURATION - CUSTOMIZE HERE
    # ================================================================   
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        

        
        # Reward for moving fast
        self.speed_reward_multiplier = 0.5  
        self.min_speed_threshold = 5        
        
        # Penalty for dying
        self.death_penalty = -50            
        
        # Penalty for standing still
        self.idle_penalty = -0.1            
        self.idle_threshold = 2             

        # Bonus for advancing far
        self.distance_bonus_multiplier = 1.5  
        
        self.prev_x_pos = 0
        self.prev_time = 400
        self.stuck_counter = 0
        self.max_x_pos = 0  
        
    def reset(self, **kwargs):
        
        obs = self.env.reset(**kwargs)
        self.prev_x_pos = 0
        self.prev_time = 400
        self.stuck_counter = 0
        self.max_x_pos = 0
        return obs
    
    def step(self, action):
        
        obs, reward, done, info = self.env.step(action)
        
        # Get information from the game
        current_x_pos = info.get('x_pos', 0)
        current_time = info.get('time', 400)
        is_dead = info.get('life', 2) < 2  # A decrease in life means death
        
        # ============================================================
        # CALCULATE CUSTOM REWARD
        # ============================================================
        custom_reward = 0
        
        # 1. REWARD FOR MOVING QUICKLY TO THE RIGHT
        x_delta = current_x_pos - self.prev_x_pos
        
        if x_delta > self.min_speed_threshold:
            # Moving fast -> reward
            speed_bonus = x_delta * self.speed_reward_multiplier
            custom_reward += speed_bonus
            self.stuck_counter = 0  # Reset counter
            
            # Additional bonus if the furthest position is reached
            if current_x_pos > self.max_x_pos:
                distance_bonus = (current_x_pos - self.max_x_pos) * self.distance_bonus_multiplier
                custom_reward += distance_bonus
                self.max_x_pos = current_x_pos
                
        elif x_delta <= self.idle_threshold:
            # 2. PENALTY FOR STANDING STILL / MOVING SLOWLY
            custom_reward += self.idle_penalty
            self.stuck_counter += 1
            
            # Heavier penalty if idle for too long
            if self.stuck_counter > 50:  # Idle > 50 frames
                custom_reward += self.idle_penalty * 2
            
            if self.stuck_counter > 100:  # Idle > 100 frames
                custom_reward += self.idle_penalty * 5
        
        # 3. HEAVY PENALTY FOR DEATH
        if is_dead:
            custom_reward += self.death_penalty
        
        # 4. SLIGHT PENALTY FOR TIME PASSING
        time_penalty = -0.01 if current_time < self.prev_time else 0
        custom_reward += time_penalty
        
        # ============================================================
        # COMBINE ORIGINAL REWARD AND CUSTOM REWARD
        # ============================================================
        final_reward = reward + custom_reward
        
        # Update tracking variables
        self.prev_x_pos = current_x_pos
        self.prev_time = current_time
        
        # Debug info (optional - comment out if not needed)
        info['custom_reward'] = custom_reward
        info['original_reward'] = reward
        info['x_delta'] = x_delta
        
        return obs, final_reward, done, info


class DetailedRewardWrapper(gym.Wrapper):   
    def __init__(self, env, config=None):
        super(DetailedRewardWrapper, self).__init__(env)
        
        
        default_config = {
            # Movement rewards
            'right_movement_bonus': 1.0,      # Bonus for moving right
            'fast_movement_bonus': 2.0,       # Bonus for moving very fast
            'fast_threshold': 10,             # Threshold to be considered "fast"
            
            # Penalties
            'death_penalty': -100,            # Penalty for dying
            'idle_penalty': -0.5,             # Penalty for being idle
            'backward_penalty': -1.0,         # Penalty for moving backward
            'time_penalty': -0.01,            # Penalty over time
            
            # Progress rewards
            'new_area_bonus': 5.0,            # Bonus for reaching a new area
            'checkpoint_bonus': 10.0,         # Bonus at checkpoints
            
            # Thresholds
            'idle_threshold': 1,              # Threshold to be considered idle
            'stuck_threshold': 50,            # Number of frames to be considered "stuck"
        }
        
        # Merge with user config
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
# HELPER FUNCTION - Create an environment with custom rewards
# ============================================================================

def create_mario_env_with_custom_reward(world, stage, reward_type='custom', reward_config=None):
    """
    Creates a Mario environment with a custom reward wrapper.
    
    Args:
        world: World number (1-8)
        stage: Stage number (1-4)
        reward_type: 'custom' or 'detailed'
        reward_config: Dict for DetailedRewardWrapper configuration (optional)
    
    Returns:
        env: The wrapped environment
    """
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from gym.wrappers import GrayScaleObservation
    from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
    
    env_name = f'SuperMarioBros-{world}-{stage}-v0'
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    # Apply custom reward wrapper BEFORE grayscale
    if reward_type == 'custom':
        env = CustomRewardWrapper(env)
    elif reward_type == 'detailed':
        env = DetailedRewardWrapper(env, reward_config)
    
    # Continue with preprocessing
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    
    return env


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CUSTOM REWARD WRAPPER - EXAMPLES")
    print("=" * 70)
    
    print("\n1. Using CustomRewardWrapper (Simple):")
    print("-" * 70)
    print("""
    from custom_rewards import create_mario_env_with_custom_reward
    
    env = create_mario_env_with_custom_reward(
        world=1, 
        stage=1, 
        reward_type='custom'
    )
    """)
    
    print("\n2. Using DetailedRewardWrapper (Advanced):")
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
    
    print("\n3. Customizing in CustomRewardWrapper:")
    print("-" * 70)
    print("""
    # Open the custom_rewards.py file and edit the values:
    
    self.speed_reward_multiplier = 1.0  # Increase for a higher reward
    self.death_penalty = -100          # Increase for a heavier penalty
    self.idle_penalty = -0.5           # Increase to penalize being idle more
    """)