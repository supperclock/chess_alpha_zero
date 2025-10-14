// libxqai.c
// Single-file minimal but practical Chinese chess engine core
// Implements: move generation (from int board), in_check, evaluate, PVS search, root interface.
//
// Compile with: gcc -O3 -fPIC -shared -o libxqai.so libxqai.c

#include "libxqai.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROWS 10
#define COLS 9
#define CELLS (ROWS*COLS)
#define MAX_DEPTH 4
#define MATE_SCORE 1000000
#define MAX_INT 1000000000

// piece codes (positive for black, negative for red)
enum {
    PC_EMPTY = 0,
    PC_ROOK = 1,
    PC_HORSE = 2,
    PC_ELEPH = 3,
    PC_ADVIS = 4,
    PC_GENERAL = 5,
    PC_CANNON = 6,
    PC_PAWN = 7
};

// base values
static const int BASEVAL[8] = {0, 900, 450, 200, 200, 10000, 400, 100};

// PST small tables
static int PST[8][ROWS][COLS];
static int pst_inited = 0;

static void init_pst(){
    if (pst_inited) return;
    pst_inited = 1;
    for (int p=0;p<8;p++) for (int y=0;y<ROWS;y++) for (int x=0;x<COLS;x++) PST[p][y][x]=0;
    for (int y=0;y<ROWS;y++){
        for (int x=0;x<COLS;x++){
            int center = (x>=3 && x<=5)?10:0;
            PST[PC_PAWN][y][x] = center + ((ROWS-1 - y) * 6);
            PST[PC_ROOK][y][x] = (x==4)?8:0;
            PST[PC_CANNON][y][x] = (y==7)?12:0;
            PST[PC_HORSE][y][x] = ((x>=3 && x<=5 && y>=3 && y<=6)?6:0);
        }
    }
}

// helpers
static inline int inside(int x,int y){ return x>=0 && x<COLS && y>=0 && y<ROWS; }
static inline int idx(int y,int x){ return y*COLS + x; }

// find general position: side_black: 1->black, 0->red
static int find_general(const int *b, int side_black, int *out_y, int *out_x){
    int code = side_black ? PC_GENERAL : -PC_GENERAL;
    for (int y=0;y<ROWS;y++){
        for (int x=0;x<COLS;x++){
            if (b[idx(y,x)] == code){
                *out_y = y; *out_x = x; return 1;
            }
        }
    }
    return 0;
}

// kings facing
static int kings_facing(const int *b){
    int ry=-1,rx=-1, by=-1,bx=-1;
    if (!find_general(b, 0, &ry, &rx)) return 0;
    if (!find_general(b, 1, &by, &bx)) return 0;
    if (rx != bx) return 0;
    int y1 = ry < by ? ry : by;
    int y2 = ry < by ? by : ry;
    for (int y=y1+1;y<y2;y++){
        if (b[idx(y,rx)] != 0) return 0;
    }
    return 1;
}

// in_check: check if side (0 black, 1 red) is in check
static int in_check(const int *b, int side){ 
    int king_y, king_x;
    if (!find_general(b, side==0?1:0, &king_y, &king_x)) return 1; // no king -> in check
    int opp_sign = side==0 ? -1 : 1; // if side black (0), opp are red (negative)
    // straight lines
    const int dirs4[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    for (int d=0;d<4;d++){
        int dx = dirs4[d][1], dy = dirs4[d][0];
        int steps = 0;
        for (int i=1;i<10;i++){
            int nx = king_x + dx*i, ny = king_y + dy*i;
            if (!inside(nx,ny)) break;
            int p = b[idx(ny,nx)];
            if (p!=0){
                int pside = p>0?1: -1;
                int is_opp = (pside == opp_sign);
                int t = p>0? p : -p;
                if (is_opp){
                    if ((t==PC_ROOK) && steps==0) return 1;
                    if ((t==PC_CANNON) && steps==1) return 1;
                    if ((t==PC_GENERAL) && steps==0) return 1;
                    if ((t==PC_PAWN) && steps==0){
                        // pawn forward or sideways check, approx: use relative direction
                        // for simplicity, allow pawn to check if adjacent forward/side depending
                        if ( (p>0 && dy== -1 && dx==0) || (p<0 && dy==1 && dx==0) ) return 1;
                    }
                }
                steps++;
                if (steps>1) break;
            }
        }
    }
    // horse attacks (approximate: check knight squares ignoring legs for speed could be wrong; implement leg check)
    const int h8[8][2] = {{1,2},{2,1},{-1,2},{-2,1},{1,-2},{2,-1},{-1,-2},{-2,-1}};
    for (int i=0;i<8;i++){
        int nx = king_x + h8[i][0], ny = king_y + h8[i][1];
        if (!inside(nx,ny)) continue;
        // compute leg
        int legx = king_x + (h8[i][0] / 2), legy = king_y;
        if (abs(h8[i][0])==1){ legx = king_x; legy = king_y + (h8[i][1] / 2); }
        if (!inside(legx,legy)) continue;
        if (b[idx(legy,legx)] != 0) continue;
        int p = b[idx(ny,nx)];
        if (p == 0) continue;
        int pside = p>0?1:-1;
        int opp_sign2 = side==0? -1 : 1;
        if (pside == opp_sign2 && ( (p>0? p : -p) == PC_HORSE )) return 1;
    }
    return 0;
}

// generate moves into a preallocated array; return count
// moves represented as 4 ints per move in out_moves: [fy,fx,ty,tx,...]
// max_moves should be >= 256
static int generate_moves_into(const int *b, int side, int *out_moves, int max_moves){
    int count = 0;
    int mysign = side==0?1:-1; // black positive
    int enemy_sign = -mysign;
    for (int y=0;y<ROWS;y++){
        for (int x=0;x<COLS;x++){
            int p = b[idx(y,x)];
            if (p==0) continue;
            if ((p>0?1:-1) != mysign) continue;
            int t = p>0? p : -p;
            if (t == PC_ROOK){
                const int dirs[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
                for (int d=0;d<4;d++){
                    int dx = dirs[d][1], dy = dirs[d][0];
                    int nx = x + dx, ny = y + dy;
                    while (inside(nx,ny)){
                        int q = b[idx(ny,nx)];
                        if (q==0){
                            if (count+4 <= max_moves){ out_moves[count++]=y; out_moves[count++]=x; out_moves[count++]=ny; out_moves[count++]=nx; }
                        } else {
                            if ((q>0?1:-1) == enemy_sign){
                                if (count+4 <= max_moves){ out_moves[count++]=y; out_moves[count++]=x; out_moves[count++]=ny; out_moves[count++]=nx; }
                            }
                            break;
                        }
                        nx += dx; ny += dy;
                    }
                }
            } else if (t == PC_CANNON){
                const int dirs[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
                for (int d=0;d<4;d++){
                    int dx = dirs[d][1], dy = dirs[d][0];
                    int nx = x + dx, ny = y + dy;
                    // free moves
                    while (inside(nx,ny) && b[idx(ny,nx)]==0){
                        if (count+4 <= max_moves){ out_moves[count++]=y; out_moves[count++]=x; out_moves[count++]=ny; out_moves[count++]=nx; }
                        nx += dx; ny += dy;
                    }
                    // find a screen
                    nx += dx; ny += dy;
                    while (inside(nx,ny)){
                        int q = b[idx(ny,nx)];
                        if (q!=0){
                            if ((q>0?1:-1) == enemy_sign){
                                if (count+4 <= max_moves){ out_moves[count++]=y; out_moves[count++]=x; out_moves[count++]=ny; out_moves[count++]=nx; }
                            }
                            break;
                        }
                        nx += dx; ny += dy;
                    }
                }
            } else if (t == PC_HORSE){
                const int h8[8][2] = {{1,2},{2,1},{-1,2},{-2,1},{1,-2},{2,-1},{-1,-2},{-2,-1}};
                for (int i=0;i<8;i++){
                    int nx = x + h8[i][0], ny = y + h8[i][1];
                    // leg
                    int legx = x + (h8[i][0]/2), legy = y;
                    if (abs(h8[i][0])==1){ legx = x; legy = y + (h8[i][1]/2); }
                    if (!inside(nx,ny) || !inside(legx,legy)) continue;
                    if (b[idx(legy,legx)] != 0) continue;
                    int q = b[idx(ny,nx)];
                    if (q==0 || (q>0?1:-1)==enemy_sign){
                        if (count+4 <= max_moves){ out_moves[count++]=y; out_moves[count++]=x; out_moves[count++]=ny; out_moves[count++]=nx; }
                    }
                }
            } else if (t == PC_PAWN){
                int forward = mysign>0? -1 : 1; // black moves up (y--)
                int ny = y + forward;
                if (inside(x,ny)){
                    int q = b[idx(ny,x)];
                    if (q==0 || (q>0?1:-1)==enemy_sign){
                        if (count+4 <= max_moves){ out_moves[count++]=y; out_moves[count++]=x; out_moves[count++]=ny; out_moves[count++]=x; }
                    }
                }
                // crossed
                int crossed = (mysign>0 && y<=4) || (mysign<0 && y>=5);
                if (crossed){
                    if (inside(x-1,y)){
                        int q = b[idx(y,x-1)];
                        if (q==0 || (q>0?1:-1)==enemy_sign){
                            if (count+4 <= max_moves){ out_moves[count++]=y; out_moves[count++]=x; out_moves[count++]=y; out_moves[count++]=x-1; }
                        }
                    }
                    if (inside(x+1,y)){
                        int q = b[idx(y,x+1)];
                        if (q==0 || (q>0?1:-1)==enemy_sign){
                            if (count+4 <= max_moves){ out_moves[count++]=y; out_moves[count++]=x; out_moves[count++]=y; out_moves[count++]=x+1; }
                        }
                    }
                }
            } else if (t == PC_ELEPH){
                const int E4[4][2] = {{2,2},{2,-2},{-2,2},{-2,-2}};
                for (int k=0;k<4;k++){
                    int nx = x + E4[k][0], ny = y + E4[k][1];
                    int mx = x + E4[k][0]/2, my = y + E4[k][1]/2;
                    if (!inside(nx,ny) || !inside(mx,my)) continue;
                    // side-specific limit
                    if (mysign>0 && ny < 5) continue;
                    if (mysign<0 && ny > 4) continue;
                    if (b[idx(my,mx)] != 0) continue;
                    int q = b[idx(ny,nx)];
                    if (q==0 || (q>0?1:-1)==enemy_sign){
                        if (count+4 <= max_moves){ out_moves[count++]=y; out_moves[count++]=x; out_moves[count++]=ny; out_moves[count++]=nx; }
                    }
                }
            } else if (t == PC_ADVIS){
                const int D4O[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
                for (int k=0;k<4;k++){
                    int nx = x + D4O[k][1], ny = y + D4O[k][0];
                    if (!inside(nx,ny)) continue;
                    if (mysign>0 && !(ny>=7 && ny<=9 && nx>=3 && nx<=5)) continue;
                    if (mysign<0 && !(ny>=0 && ny<=2 && nx>=3 && nx<=5)) continue;
                    int q = b[idx(ny,nx)];
                    if (q==0 || (q>0?1:-1)==enemy_sign){
                        if (count+4 <= max_moves){ out_moves[count++]=y; out_moves[count++]=x; out_moves[count++]=ny; out_moves[count++]=nx; }
                    }
                }
            } else if (t == PC_GENERAL){
                const int D4[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
                for (int k=0;k<4;k++){
                    int nx = x + D4[k][1], ny = y + D4[k][0];
                    if (!inside(nx,ny)) continue;
                    if (mysign>0 && !(ny>=7 && ny<=9 && nx>=3 && nx<=5)) continue;
                    if (mysign<0 && !(ny>=0 && ny<=2 && nx>=3 && nx<=5)) continue;
                    int q = b[idx(ny,nx)];
                    if (q==0 || (q>0?1:-1)==enemy_sign){
                        if (count+4 <= max_moves){ out_moves[count++]=y; out_moves[count++]=x; out_moves[count++]=ny; out_moves[count++]=nx; }
                    }
                }
                // face opponent general
                int oy, ox;
                if (find_general(b, mysign<0?1:0, &oy, &ox) && ox == x){
                    int blocked = 0;
                    int y1 = y < oy ? y : oy;
                    int y2 = y < oy ? oy : y;
                    for (int yy=y1+1; yy<y2; yy++){
                        if (b[idx(yy,x)] != 0){ blocked = 1; break; }
                    }
                    if (!blocked){
                        if (count+4 <= max_moves){ out_moves[count++]=y; out_moves[count++]=x; out_moves[count++]=oy; out_moves[count++]=ox; }
                    }
                }
            }
        }
    }
    return count/4;
}

// make/unmake on board array (caller supplies mutable int board90)
// make returns captured value (int)
static int make_move_on_board(int *b, int fy, int fx, int ty, int tx){
    int from = b[idx(fy,fx)];
    int cap = b[idx(ty,tx)];
    b[idx(ty,tx)] = from;
    b[idx(fy,fx)] = 0;
    return cap;
}
static void unmake_move_on_board(int *b, int fy, int fx, int ty, int tx, int captured){
    int moving = b[idx(ty,tx)];
    b[idx(fy,fx)] = moving;
    b[idx(ty,tx)] = captured;
}

// evaluate fast (black perspective positive)
int xq_evaluate(const int *board90){
    init_pst();
    long mat_black=0, mat_red=0, pst_black=0, pst_red=0;
    for (int y=0;y<ROWS;y++){
        for (int x=0;x<COLS;x++){
            int v = board90[idx(y,x)];
            if (v==0) continue;
            int av = v>0? v : -v;
            if (av<0 || av>=8) continue;
            int base = BASEVAL[av];
            int pstv = PST[av][y][x];
            if (v>0){ mat_black += base; pst_black += pstv; }
            else { mat_red += base; pst_red += pstv; }
        }
    }
    long score = (mat_black - mat_red) + (pst_black - pst_red);
    // simple king safety
    int ky,kx;
    if (!find_general(board90, 1, &ky, &kx)) return -900000;
    // compute palace threats
    int palace_val_black = 0;
    for (int yy=7; yy<=9; yy++){
        for (int xx=3; xx<=5; xx++){
            int c = board90[idx(yy,xx)];
            if (c<0) palace_val_black += BASEVAL[-c];
        }
    }
    score -= (palace_val_black/20)*18;
    if (!find_general(board90, 0, &ky, &kx)) return 900000;
    int palace_val_red = 0;
    for (int yy=0; yy<=2; yy++){
        for (int xx=3; xx<=5; xx++){
            int c = board90[idx(yy,xx)];
            if (c>0) palace_val_red += BASEVAL[c];
        }
    }
    score += (palace_val_red/20)*18;
    if (score > 1000000) score = 1000000;
    if (score < -1000000) score = -1000000;
    return (int)score;
}

// Quiescence (very small): only captures
static int quiescence_search(int *board, int alpha, int beta, int side){
    int stand = xq_evaluate((const int*)board);
    if (stand >= beta) return beta;
    if (stand > alpha) alpha = stand;
    // generate captures only
    int moves_buf[1024];
    int nm = generate_moves_into(board, side, moves_buf, 1024);
    for (int i=0;i<nm;i++){
        int fy = moves_buf[i*4+0], fx = moves_buf[i*4+1], ty = moves_buf[i*4+2], tx = moves_buf[i*4+3];
        if (board[idx(ty,tx)] == 0) continue;
        int cap = make_move_on_board(board, fy, fx, ty, tx);
        int val = -quiescence_search(board, -beta, -alpha, 1-side);
        unmake_move_on_board(board, fy, fx, ty, tx, cap);
        if (val >= beta) return beta;
        if (val > alpha) alpha = val;
    }
    return alpha;
}

// PVS negamax
static int pvs_rec(int *board, int depth, int alpha, int beta, int side){
    if (depth <= 0) return quiescence_search(board, alpha, beta, side);
    if (in_check((const int*)board, side)) {
        // don't null-move if in check
    } else if (depth >= 3){
        // null-move reduction
        int R = 2;
        int val = -pvs_rec(board, depth-1-R, -beta, -beta+1, 1-side);
        if (val >= beta) return beta;
    }
    int moves_buf[4096];
    int nm = generate_moves_into(board, side, moves_buf, 4096);
    if (nm==0){
        if (in_check((const int*)board, side)) return -MATE_SCORE + depth;
        return 0;
    }
    int best = -MAX_INT;
    int first = 1;
    for (int i=0;i<nm;i++){
        int fy = moves_buf[i*4+0], fx = moves_buf[i*4+1], ty = moves_buf[i*4+2], tx = moves_buf[i*4+3];
        int captured = make_move_on_board(board, fy, fx, ty, tx);
        int val;
        if (first){
            val = -pvs_rec(board, depth-1, -beta, -alpha, 1-side);
        } else {
            val = -pvs_rec(board, depth-1, -alpha-1, -alpha, 1-side);
            if (val > alpha && val < beta) val = -pvs_rec(board, depth-1, -beta, -alpha, 1-side);
        }
        unmake_move_on_board(board, fy, fx, ty, tx, captured);
        if (val > best) best = val;
        if (val > alpha) alpha = val;
        if (alpha >= beta) break;
        first = 0;
    }
    return best;
}

// root entry: boards are passed as const int*, copy to local mutable buffer
int xq_minimax_root(const int *board90, int side, double time_limit, int *out_move){
    init_pst();
    int board_buf[CELLS];
    memcpy(board_buf, board90, sizeof(int)*CELLS);
    int best_fy=0,best_fx=0,best_ty=0,best_tx=0;
    int best_val = -MAX_INT;
    clock_t start = clock();
    // iterative deepening up to MAX_DEPTH
    for (int depth=1; depth<=MAX_DEPTH; depth++){
        int moves_buf[4096];
        int nm = generate_moves_into(board_buf, side, moves_buf, 4096);
        if (nm==0) break;
        int alpha = -MATE_SCORE, beta = MATE_SCORE;
        for (int i=0;i<nm;i++){
            // check time
            double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
            if (time_limit > 0.0 && elapsed > time_limit) goto time_over;
            int fy = moves_buf[i*4+0], fx = moves_buf[i*4+1], ty = moves_buf[i*4+2], tx = moves_buf[i*4+3];
            int captured = make_move_on_board(board_buf, fy, fx, ty, tx);
            int val = -pvs_rec(board_buf, depth-1, -beta, -alpha, 1-side);
            unmake_move_on_board(board_buf, fy, fx, ty, tx, captured);
            if (val > best_val){
                best_val = val;
                best_fy = fy; best_fx = fx; best_ty = ty; best_tx = tx;
            }
            if (val > alpha) alpha = val;
            if (alpha >= beta) break;
        }
        // if time nearly up, break
        double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
        if (time_limit > 0.0 && elapsed > time_limit) break;
    }
time_over:
    out_move[0]=best_fy; out_move[1]=best_fx; out_move[2]=best_ty; out_move[3]=best_tx;
    return 1;
}

void xq_init(){ init_pst(); }
