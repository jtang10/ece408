dimGrid(H_out * W_out / TILE_WIDTH, M / TILE_WIDTH, B);
dimBlock(TILE_WIDTH, TILE_WIDTH);

row = by * TILE_WIDTH + ty;
col = bx * TILE_WIDTH + tx;
numMatAColumn = C * K * K;

float acc = 0

for (int i = 0;)