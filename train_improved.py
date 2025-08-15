"""
改进的训练脚本
"""
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import logging
from datetime import datetime

from config import TrainingConfig, DEFAULT_CONFIG
from env_wrapper import make_env, state_to_tensor
from model import PolicyValueNet
from mcts import MCTS
from replay_buffer import ReplayBuffer
from utils import set_random_seed, save_config, create_checkpoint, get_model_summary

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_epoch(env, net, mcts, buffer, config, device):
    """训练一个epoch"""
    try:
        # 重置环境
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        states, mcts_probs, rewards = [], [], []
        done = False
        step_count = 0
        
        while not done and step_count < config.max_steps_per_game:
            try:
                # 转换状态
                tensor = state_to_tensor(obs)
                
                # MCTS搜索
                probs = mcts.play(env)
                
                # 归一化概率
                if np.sum(probs) > 0:
                    probs = probs / np.sum(probs)
                else:
                    probs = np.ones(len(probs)) / len(probs)
                
                # 选择动作
                action = np.random.choice(range(env.action_space.n), p=probs)
                
                # 存储经验
                states.append(tensor)
                mcts_probs.append(probs)
                
                # 执行动作
                step_result = env.step(action)
                if len(step_result) == 4:  # 旧版本gym
                    obs, reward, done, _ = step_result
                else:  # 新版本gym
                    obs, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                
                rewards.append(reward)
                step_count += 1
                
            except Exception as e:
                logger.warning(f"Error during game step: {e}")
                break
        
        # 将经验添加到缓冲区
        for s, p, r in zip(states, mcts_probs, rewards):
            buffer.add(s, p, r)
        
        return len(states), step_count
        
    except Exception as e:
        logger.error(f"Error in training epoch: {e}")
        return 0, 0

def train_step(net, buffer, optimizer, config, device):
    """执行一步训练"""
    try:
        batch = buffer.sample(config.batch_size)
        if not batch:
            return None
        
        state_b, prob_b, reward_b = zip(*batch)
        state_b = torch.tensor(np.array(state_b), dtype=torch.float32).to(device)
        prob_b = torch.tensor(np.array(prob_b), dtype=torch.float32).to(device)
        reward_b = torch.tensor(reward_b, dtype=torch.float32).to(device)
        
        # 前向传播
        logp, value = net(state_b)
        
        # 计算损失
        policy_loss = -torch.mean(torch.sum(prob_b * logp, dim=1))
        value_loss = torch.mean((reward_b - value.squeeze())**2)
        total_loss = policy_loss + value_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
        
    except Exception as e:
        logger.error(f"Error in training step: {e}")
        return None

def main():
    """主训练函数"""
    # 加载配置
    config = DEFAULT_CONFIG
    logger.info("Starting training with configuration:")
    logger.info(f"Device: {config.device}")
    logger.info(f"Board size: {config.board_height}x{config.board_width}")
    logger.info(f"MCTS simulations: {config.mcts_simulations}")
    
    # 设置随机种子
    set_random_seed(42)
    
    # 保存配置
    save_config(config, "training_config.json")
    
    try:
        # 创建环境
        env = make_env()
        action_size = env.action_space.n
        logger.info(f"Environment created. Action space size: {action_size}")
        
        # 创建模型
        net = PolicyValueNet(
            action_size=action_size,
            board_height=config.board_height,
            board_width=config.board_width
        ).to(config.device)
        
        logger.info("Model created:")
        logger.info(get_model_summary(net))
        
        # 创建优化器
        optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
        
        # 创建缓冲区
        buffer = ReplayBuffer(capacity=config.replay_buffer_capacity)
        
        # 创建MCTS
        def policy_value_fn(obs):
            try:
                x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(config.device)
                with torch.no_grad():
                    logp, v = net(x)
                return torch.exp(logp[0]).cpu().numpy(), v.item()
            except Exception as e:
                logger.warning(f"Error in policy_value_fn: {e}")
                return np.ones(action_size) / action_size, 0.0
        
        mcts = MCTS(
            policy_value_fn=policy_value_fn,
            c_puct=config.mcts_c_puct,
            sims=config.mcts_simulations
        )
        
        # 训练循环
        best_loss = float('inf')
        training_history = []
        
        logger.info("Starting training loop...")
        
        for epoch in range(config.num_epochs):
            try:
                # 训练一个epoch
                num_states, num_steps = train_epoch(env, net, mcts, buffer, config, config.device)
                
                if num_states > 0:
                    # 执行训练步骤
                    loss_info = train_step(net, buffer, optimizer, config, config.device)
                    
                    if loss_info:
                        current_loss = loss_info['total_loss']
                        training_history.append({
                            'epoch': epoch,
                            'loss': current_loss,
                            'policy_loss': loss_info['policy_loss'],
                            'value_loss': loss_info['value_loss'],
                            'num_states': num_states,
                            'num_steps': num_steps
                        })
                        
                        # 打印进度
                        logger.info(
                            f"Epoch {epoch+1}/{config.num_epochs} - "
                            f"Loss: {current_loss:.4f} "
                            f"(Policy: {loss_info['policy_loss']:.4f}, "
                            f"Value: {loss_info['value_loss']:.4f}) - "
                            f"States: {num_states}, Steps: {num_steps}"
                        )
                        
                        # 保存最佳模型
                        if current_loss < best_loss:
                            best_loss = current_loss
                            torch.save(net.state_dict(), os.path.join(config.model_save_dir, "best_model.pth"))
                            logger.info(f"New best model saved with loss: {best_loss:.4f}")
                        
                        # 定期保存检查点
                        if (epoch + 1) % config.save_interval == 0:
                            checkpoint_path = os.path.join(
                                config.model_save_dir, 
                                f"checkpoint_epoch_{epoch+1}.pth"
                            )
                            create_checkpoint(net, optimizer, epoch, current_loss, checkpoint_path)
                
                # 打印缓冲区统计
                if epoch % 10 == 0:
                    stats = buffer.get_stats()
                    logger.info(f"Buffer stats: {stats}")
                
            except Exception as e:
                logger.error(f"Error in epoch {epoch}: {e}")
                continue
        
        # 保存最终模型
        final_model_path = os.path.join(config.model_save_dir, "final_model.pth")
        torch.save(net.state_dict(), final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        # 保存训练历史
        import json
        with open("training_history.json", 'w') as f:
            json.dump(training_history, f, indent=2)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Critical error during training: {e}")
        raise

if __name__ == "__main__":
    main()
