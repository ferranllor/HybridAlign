# Builds the correctness harness against the project sources.
#
#   make -C tools                       -> tools/bin/verify_dp with the default BLOCKSIZE
#   make -C tools BLOCKSIZE=64          -> same, but every kernel compiled with BLOCKSIZE=64
#   make -C tools clean
#
# Everything is compiled into tools/obj so the main build in obj/ is never disturbed.

CCNV      = nvcc
ROOT      = ..
INCLUDES  = -I$(ROOT) -I$(ROOT)/include
FLAGS     = -lm -Xcompiler -fopenmp -arch=sm_120 -diag-suppress 177 -diag-suppress 20054

BLOCKSIZE ?=
N_BUFFERS ?=
ifneq ($(BLOCKSIZE),)
FLAGS += -DBLOCKSIZE=$(BLOCKSIZE)
endif
ifneq ($(N_BUFFERS),)
FLAGS += -DN_BUFFERS=$(N_BUFFERS)
endif

OBJ_DIR = obj
BIN_DIR = bin

CU_SRC  = cuda_naive cuda_parallel_node cuda_parallel_async cuda_async_monolithic \
          cuda_async_batching cuda_shared_mem hybrid_base hybrid_unified
CU_OBJ  = $(patsubst %,$(OBJ_DIR)/%.o,$(CU_SRC))
C_OBJ   = $(OBJ_DIR)/cpu_sequential.o $(OBJ_DIR)/cpu_simd_parallel_node.o

all: $(BIN_DIR)/verify_dp

$(OBJ_DIR)/%.o: $(ROOT)/src/%.cu
	@mkdir -p $(OBJ_DIR)
	$(CCNV) -c $< -o $@ $(FLAGS) $(INCLUDES)

$(OBJ_DIR)/%.o: $(ROOT)/src/%.c
	@mkdir -p $(OBJ_DIR)
	gcc -c $< -o $@ -lm -Ofast -finline-functions -fopenmp $(INCLUDES)

$(BIN_DIR)/verify_dp: verify_dp.cu $(CU_OBJ) $(C_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CCNV) $^ -o $@ $(FLAGS) $(INCLUDES)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean
