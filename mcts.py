import math
import numpy as np
import copy
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCTSNode:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def expand(self, actions, priors):
        """扩展节点"""
        for action, prior in zip(actions, priors):
            if action not in self.children:
                self.children[action] = MCTSNode(prior=prior)
        self.is_expanded = True

class MCTS:
    def __init__(self, policy_value_fn, c_puct=1.0, sims=200):
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.sims = sims

    def play(self, env):
        try:
            root = MCTSNode(0)
            
            for _ in range(self.sims):
                try:
                    # 创建环境的深度复制
                    env_copy = copy.deepcopy(env)
                    node = root
                    path = [node]
                    
                    # Selection
                    while node.children and not env_copy.done:
                        total = sum(child.visit_count for child in node.children.values())
                        action, node = max(
                            node.children.items(),
                            key=lambda item: item[1].value() + 
                                           self.c_puct * item[1].prior * 
                                           math.sqrt(total) / (1 + item[1].visit_count)
                        )
                        
                        # 执行动作
                        try:
                            step_result = env_copy.step(action)
                            if len(step_result) == 4:  # 旧版本gym
                                obs, reward, done, _ = step_result
                            else:  # 新版本gym
                                obs, reward, terminated, truncated, _ = step_result
                                done = terminated or truncated
                            
                            if done:
                                break
                            path.append(node)
                        except Exception as e:
                            logger.warning(f"Error during MCTS step: {e}")
                            break
                    
                    # Expansion and evaluation
                    if not env_copy.done:
                        try:
                            obs = env_copy.observation
                            if obs is None:
                                # 尝试获取当前状态
                                obs = env_copy._get_obs()
                            
                            if obs is not None:
                                policy, value = self.policy_value_fn(obs)
                                
                                # 获取合法动作
                                legal_actions = self._get_legal_actions(env_copy)
                                
                                if not node.is_expanded and legal_actions:
                                    # 扩展节点
                                    legal_policies = [policy[a] for a in legal_actions]
                                    node.expand(legal_actions, legal_policies)
                                
                                # 归一化概率
                                if np.sum(policy) > 0:
                                    policy = policy / np.sum(policy)
                                else:
                                    policy = np.ones(len(policy)) / len(policy)
                            else:
                                value = 0.0
                        except Exception as e:
                            logger.warning(f"Error during expansion: {e}")
                            value = 0.0
                    else:
                        # 游戏结束，获取奖励
                        try:
                            value = env_copy._get_reward()
                        except:
                            value = 0.0
                    
                    # Backpropagation
                    for node in reversed(path):
                        node.visit_count += 1
                        node.value_sum += value
                        value = -value  # 对手的视角
                        
                except Exception as e:
                    logger.warning(f"Error in MCTS simulation: {e}")
                    continue
            
            # 返回动作概率
            if root.children:
                counts = np.array([root.children[a].visit_count if a in root.children else 0 
                                 for a in range(env.action_space.n)])
                total = counts.sum()
                if total > 0:
                    probs = counts / total
                else:
                    probs = np.ones(env.action_space.n) / env.action_space.n
            else:
                probs = np.ones(env.action_space.n) / env.action_space.n
            
            return probs
            
        except Exception as e:
            logger.error(f"Critical error in MCTS: {e}")
            # 返回均匀分布作为fallback
            return np.ones(env.action_space.n) / env.action_space.n
    
    def _get_legal_actions(self, env):
        """获取合法动作"""
        try:
            # 尝试不同的方法获取合法动作
            if hasattr(env, 'legal_move_mask'):
                mask = env.legal_move_mask()
                return [i for i, legal in enumerate(mask) if legal]
            elif hasattr(env, 'get_legal_moves'):
                return env.get_legal_moves()
            elif hasattr(env, 'legal_moves'):
                return env.legal_moves
            else:
                # 如果没有合法动作检查，返回所有动作
                return list(range(env.action_space.n))
        except Exception as e:
            logger.warning(f"Error getting legal actions: {e}")
            return list(range(env.action_space.n))
