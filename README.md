# 🎮 Super Mario Bros AI Training Project

This is a fascinating project that applies one of the most advanced techniques in artificial intelligence—Deep Reinforcement Learning—to teach a machine how to conquer the classic game of Super Mario Bros. on its own. The goal isn't just to create a scripted "bot" that follows predefined rules, but to build an AI agent capable of learning, making its own decisions, and developing strategies from scratch, purely by observing the screen and receiving feedback in the form of rewards.

**Why Super Mario Bros.?**

Super Mario Bros. is an ideal environment for Reinforcement Learning because it possesses all the necessary elements: a clear objective (rescue the princess), an integrated reward system (the score), and countless challenges that require a mix of quick reflexes and long-term planning. Training an AI for this game is not just an interesting technical problem; it's also a powerful visual demonstration of a machine's ability to learn and master complex virtual worlds.

In essence, this project transforms a childhood game into an AI laboratory, where we can witness a digital entity learn to master a virtual world through the iterative process of trial, failure, and success.
## Demo 
![world 1 stage 1](/demo%20gif/1-1.gif)
## 🖥️ System Requirements

- **Python**: 3.8.2
- **CUDA**: 11.8 (if using GPU)
- **RAM**: Minimum 8GB
- **Hard Drive**: 5GB free for checkpoints and videos


## 🚀 Installation

### Step 1: Clone or download the project

```bash
git clone https://github.com/Hieuabssy/mario-rl-ppo-agent.git
cd mario-rl-ppo-agent
```
---


### Step 2: Create virtual environment

```bash
#Windows
python -m venv mario_env
mario_env\Scripts\activate
#Conda
conda create -n mario_env python=3.8.2
conda activate mario_myenv
```

### Step 3: Install depedency
```bash
pip install -r requirements.txt
```



---

## 📁 Directory Structure

```
mario-ai-training/
├── train_mario.py        # Training script
├── resume_training.py    # Training from checkpoint      
├── test_mario.py         # Testing script
├── requirements.txt      # Dependencies
├── README.md             # This file
├── train/                # Directory for checkpoints
│   ├── 1-1/              # Checkpoints for level 1-1
│   ├── 1-2/              # Checkpoints for level 1-2
│   └── ...
├── logs/                  # TensorBoard logs
│   ├── 1-1/
│   ├── 1-2/
│   └── ...
├── videos/                # Test videos
│   ├── 1-1/
│   ├── 1-2/
│   └── ...
└── demo gif/              # demo gif
    ├──1-1.gif/
    ├──1-2.gift/
    ├──
```

**Note:** The train/, logs/, and videos/ directories will be created automatically when the scripts are run.

---
## 📁 Pretraipretrained model

You can see my pretrained model in [my driver](https://drive.google.com/drive/folders/1owe7zEqwn0BAEuB6SB2hXMOrWOcLzgSR)
## 📖 How to Use

### 1️⃣ Train the Model

#### Step 1: Select the level

Open the `train_mario.py` file and edit::

```python
WORLD = 1  # Select world (1-8)
STAGE = 1  # Select stage (1-4)
```

#### Step 2: Customize training parameters (optional)

```python
TOTAL_TIMESTEPS = 1000000  # Total training steps
CHECKPOINT_FREQ = 10000    # Save checkpoint every 10,000 steps
LEARNING_RATE = 0.000001   # Learning rate
N_STEPS = 512              # Number of steps per update
```

#### Step 3: Run training

```bash
python train_mario.py
```

**Sample Output:**
```
======================================================================
🎮 TRAINING AI FOR SUPER MARIO BROS - LEVEL 1-1
======================================================================
📁 Checkpoint: ./train/1-1/
📊 Logs: ./logs/1-1/
🎬 Videos: ./videos/1-1/
======================================================================

🌍 Creating environment: SuperMarioBros-1-1-v0
✅ Environment created and preprocessed successfully!

🤖 Initializing PPO model...
✅ Model initialized!

🎓 Starting training with 1,000,000 timesteps...
```

**Recommend TOTAL_TIMESTEPS=3M to 5M for each level**


### 2️⃣ Model Testing

#### **Step 1:** Test configuration

Open the file `test_mario.py` and edit:

```python
WORLD = 1  # Must match the trained screen
STAGE = 1  # Must match the trained screen

# Optional
RECORD_VIDEO = True   # Record video
NUM_EPISODES = 3      # Number of plays
RENDER_GAME = True    # Display
```

#### **Step 2:** Run test

```bash
python test_mario.py
```

#### **Step 3:** Select model

The script will display a list of available models:

EXAPMLE:
```
📋 AVAILABLE MODELS:
----------------------------------------------------------------------
1. best_model_10000 (5.2 MB)
2. best_model_20000 (5.2 MB)
3. best_model_100000 (5.2 MB)
4. final_model_1-1 (5.2 MB)
----------------------------------------------------------------------

Select model (1-4) or Enter to select latest:
```

## ⚙️  **I will update new level soon**
