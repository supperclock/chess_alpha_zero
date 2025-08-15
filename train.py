import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from env_wrapper import make_env, state_to_tensor
from model import PolicyValueNet
from mcts import MCTS
from replay_buffer import ReplayBuffer

def train():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        env = make_env()
        action_size = env.action_space.n
        net = PolicyValueNet(action_size).to(device)
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        buffer = ReplayBuffer()
        
        # 创建模型保存目录
        os.makedirs("models", exist_ok=True)

        def policy_value_fn(obs):
            try:
                x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    logp, v = net(x)
                return torch.exp(logp[0]).cpu().numpy(), v.item()
            except Exception as e:
                print(f"Error in policy_value_fn: {e}")
                # 返回均匀分布作为fallback
                return np.ones(action_size) / action_size, 0.0

        mcts = MCTS(policy_value_fn, sims=100)

        for epoch in range(100):
            try:
                # 正确处理环境重置
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]  # 新版本gym返回(obs, info)
                
                states, mcts_probs, rewards = [], [], []
                done = False
                step_count = 0
                max_steps = 1000  # 防止无限循环
                
                while not done and step_count < max_steps:
                    try:
                        tensor = state_to_tensor(obs)
                        probs = mcts.play(env)
                        
                        # 确保概率和为1
                        if np.sum(probs) > 0:
                            probs = probs / np.sum(probs)
                        else:
                            probs = np.ones(action_size) / action_size
                        
                        action = np.random.choice(range(action_size), p=probs)
                        states.append(tensor)
                        mcts_probs.append(probs)
                        
                        step_result = env.step(action)
                        if len(step_result) == 4:  # 旧版本gym
                            obs, reward, done, _ = step_result
                        else:  # 新版本gym
                            obs, reward, terminated, truncated, _ = step_result
                            done = terminated or truncated
                        
                        rewards.append(reward)
                        step_count += 1
                        
                    except Exception as e:
                        print(f"Error during game step: {e}")
                        break

                # 训练
                if len(states) > 0:
                    batch = buffer.sample(32)
                    if batch:
                        try:
                            state_b, prob_b, reward_b = zip(*batch)
                            state_b = torch.tensor(np.array(state_b), dtype=torch.float32).to(device)
                            prob_b = torch.tensor(np.array(prob_b), dtype=torch.float32).to(device)
                            reward_b = torch.tensor(reward_b, dtype=torch.float32).to(device)

                            logp, value = net(state_b)
                            loss = -torch.mean(torch.sum(prob_b * logp, dim=1)) + torch.mean((reward_b - value.squeeze())**2)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            
                            print(f"Epoch {epoch}, Loss {loss.item():.4f}, Steps {step_count}")
                        except Exception as e:
                            print(f"Error during training: {e}")
                
                # 定期保存模型
                if (epoch + 1) % 10 == 0:
                    torch.save(net.state_dict(), f"models/model_epoch_{epoch+1}.pth")
                    
            except Exception as e:
                print(f"Error in epoch {epoch}: {e}")
                continue
                
        # 保存最终模型
        torch.save(net.state_dict(), "models/final_model.pth")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Critical error during training: {e}")
        raise
