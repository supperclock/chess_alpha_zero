from opening_book import fen_to_board
import ai

fen = '2EA1A3/4K3h/2H6/P7P/2P3p2/3P5/p7p/4c4/6H2/2eakae2 b - - 0 1'
# fen_to_board in opening_book assumes red is on top (uppercase for red)
board = fen_to_board(fen)

print('Board loaded. Black to move.')
print('Black in check?', ai.in_check(board, 'black'))

moves = ai.generate_moves(board, 'black')
print('Generated moves for black:', len(moves))

results = []
for m in moves:
    cap = ai.make_move(board, m)
    still_in_check = ai.in_check(board, 'black')
    results.append((m.to_dict(), not still_in_check))
    ai.unmake_move(board, m, cap)

for i, (mv, ok) in enumerate(results):
    print(i, mv, 'evade_check=', ok)

print('Total evasions:', sum(1 for _,ok in results if ok), '/', len(results))
