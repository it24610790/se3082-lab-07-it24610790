CC = mpicc
CFLAGS = -O2

SRCS = sum_bcast.c sum_scatter.c sum_gather.c sum_reduce.c sum_allreduce.c sum_scan
BINS = sum_bcast sum_scatter sum_gather sum_reduce sum_allreduce sum_scan

all: $(BINS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

run: all
	@echo "\n=== Running sum_bcast ==="
	mpirun --oversubscribe -np 4 ./sum_bcast
	@echo "\n=== Running sum_scatter ==="
	mpirun --oversubscribe -np 4 ./sum_scatter
	@echo "\n=== Running sum_gather ==="
	mpirun --oversubscribe -np 4 ./sum_gather
	@echo "\n=== Running sum_reduce ==="
	mpirun --oversubscribe -np 4 ./sum_reduce
	@echo "\n=== Running sum_allreduce ==="
	mpirun --oversubscribe -np 4 ./sum_allreduce
	@echo "\n=== Running sum_scan ==="
	mpirun --oversubscribe -np 4 ./sum_scan

clean:
	rm -f $(BINS)
