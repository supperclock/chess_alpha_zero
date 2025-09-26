# evaluate_models.py
import torch
import random
from alphazero_xiangqi import (
    AlphaNet, MCTS, initial_board, apply_move, generate_legal_moves,
    board_to_tensor, is_terminal
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def play_game(net_red, net_black, sims=50, max_moves=200):
    """
    用 net_red vs net_black 进行一盘棋
    返回: +1=红胜, -1=黑胜, 0=和棋
    """
    board = initial_board()
    side = 'red'
    terminal = False
    result = 0.0

    # 初始化 MCTS
    mcts_red = MCTS(net_red, sims=sims, device=DEVICE)
    mcts_black = MCTS(net_black, sims=sims, device=DEVICE)

    for turn in range(max_moves):
        mcts = mcts_red if side == 'red' else mcts_black
        move, _ = mcts.select_move(board, side, temperature=0.0)
        if move is None:
            # 当前方无路可走 → 负
            result = -1.0 if side == 'red' else 1.0
            return result
        board, _ = apply_move(board, move)
        side = 'red' if side == 'black' else 'black'

        terminal, z = is_terminal(board, side)
        if terminal:
            result = z
            return result

    return 0.0  # 超过最大步数，判和


def evaluate_models(net_new, net_best, games=100, sims=50, threshold=0.55):
    """
    让 net_new vs net_best 对战 N 局，统计胜率并判断是否替换 best
    """
    wins, losses, draws = 0, 0, 0

    for g in range(games):
        # 交替执先：偶数局 net_new 红，奇数局 net_new 黑
        if g % 2 == 0:
            result = play_game(net_new, net_best, sims=sims)
        else:
            result = -play_game(net_best, net_new, sims=sims)

        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

        print(f"Game {g+1}/{games}: result={result} (累积 W:{wins} D:{draws} L:{losses})")

    total = wins + losses + draws
    winrate = wins / total
    print(f"\n最终统计: W:{wins} D:{draws} L:{losses} (winrate={winrate:.3f})")

    if winrate > threshold:
        print("✅ net_new 胜率超过阈值，替换 best_model")
        torch.save(net_new.state_dict(), "best_model.pth")
        return True, winrate
    else:
        print("❌ 胜率不足，保留原 best_model")
        return False, winrate


if __name__ == "__main__":
    # 加载模型
    net_new = AlphaNet()
    net_best = AlphaNet()

    net_new.load_state_dict(torch.load("net_new.pt", map_location=DEVICE))
    net_best.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))

    net_new.to(DEVICE)
    net_best.to(DEVICE)

    # 运行评估
    evaluate_models(net_new, net_best, games=100, sims=50, threshold=0.55)
