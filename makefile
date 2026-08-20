CC=gcc
CCNV=nvcc

LIBS=
INCLUDES=-I../../ -Iinclude
LIB_FLAGS=-lm -Ofast -finline-functions -fopenmp

INCLUDES_NV=-I../../ -Iinclude
LIB_FLAGS_NV=-lm -Xcompiler -fopenmp -arch=sm_120 $(EXTRA)

BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
BATCH_OUT_FOLDER := outputs


MAIN_NAME=main
MAIN_BIN=$(MAIN_NAME)
MAIN_SRC=$(MAIN_NAME).c

OBJECTS = $(OBJ_FOLDER)/my_time_lib.o $(OBJ_FOLDER)/cpu_sequential.o $(OBJ_FOLDER)/cpu_simd.o $(OBJ_FOLDER)/cpu_simd_parallel_dp.o $(OBJ_FOLDER)/cpu_simd_parallel_node.o $(OBJ_FOLDER)/cuda_naive.o $(OBJ_FOLDER)/cuda_parallel_node.o $(OBJ_FOLDER)/cuda_parallel_async.o $(OBJ_FOLDER)/cuda_async_monolithic.o $(OBJ_FOLDER)/cuda_async_batching.o $(OBJ_FOLDER)/cuda_shared_mem.o $(OBJ_FOLDER)/cuda_persistent_kernels.o $(OBJ_FOLDER)/hybrid_base.o $(OBJ_FOLDER)/hybrid_unified.o $(OBJ_FOLDER)/hybrid_pinned.o $(OBJ_FOLDER)/cuda_no_copy.o

all: $(BIN_FOLDER)/$(MAIN_BIN)

$(OBJ_FOLDER)/my_time_lib.o: $(SRC_FOLDER)/my_time_lib.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/my_time_lib.c -o $@ $(LIB_FLAGS) $(INCLUDES)

$(OBJ_FOLDER)/cpu_sequential.o: $(SRC_FOLDER)/cpu_sequential.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/cpu_sequential.c -o $@ $(LIB_FLAGS) $(INCLUDES)

$(OBJ_FOLDER)/cpu_simd.o: $(SRC_FOLDER)/cpu_simd.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/cpu_simd.c -o $@ $(LIB_FLAGS) $(INCLUDES)

$(OBJ_FOLDER)/cpu_simd_parallel_dp.o: $(SRC_FOLDER)/cpu_simd_parallel_dp.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/cpu_simd_parallel_dp.c -o $@ $(LIB_FLAGS) $(INCLUDES)

$(OBJ_FOLDER)/cpu_simd_parallel_node.o: $(SRC_FOLDER)/cpu_simd_parallel_node.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/cpu_simd_parallel_node.c -o $@ $(LIB_FLAGS) $(INCLUDES)

$(OBJ_FOLDER)/cuda_naive.o: $(SRC_FOLDER)/cuda_naive.cu
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CCNV) -c $(SRC_FOLDER)/cuda_naive.cu -o $@ $(LIB_FLAGS_NV) $(INCLUDES_NV)

$(OBJ_FOLDER)/cuda_parallel_node.o: $(SRC_FOLDER)/cuda_parallel_node.cu
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CCNV) -c $(SRC_FOLDER)/cuda_parallel_node.cu -o $@ $(LIB_FLAGS_NV) $(INCLUDES_NV)

$(OBJ_FOLDER)/cuda_parallel_async.o: $(SRC_FOLDER)/cuda_parallel_async.cu
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CCNV) -c $(SRC_FOLDER)/cuda_parallel_async.cu -o $@ $(LIB_FLAGS_NV) $(INCLUDES_NV)

$(OBJ_FOLDER)/cuda_async_monolithic.o: $(SRC_FOLDER)/cuda_async_monolithic.cu
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CCNV) -c $(SRC_FOLDER)/cuda_async_monolithic.cu -o $@ $(LIB_FLAGS_NV) $(INCLUDES_NV)

$(OBJ_FOLDER)/cuda_async_batching.o: $(SRC_FOLDER)/cuda_async_batching.cu
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CCNV) -c $(SRC_FOLDER)/cuda_async_batching.cu -o $@ $(LIB_FLAGS_NV) $(INCLUDES_NV)

$(OBJ_FOLDER)/cuda_shared_mem.o: $(SRC_FOLDER)/cuda_shared_mem.cu
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CCNV) -c $(SRC_FOLDER)/cuda_shared_mem.cu -o $@ $(LIB_FLAGS_NV) $(INCLUDES_NV)

$(OBJ_FOLDER)/cuda_persistent_kernels.o: $(SRC_FOLDER)/cuda_persistent_kernels.cu
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CCNV) -c $(SRC_FOLDER)/cuda_persistent_kernels.cu -o $@ $(LIB_FLAGS_NV) $(INCLUDES_NV)

$(OBJ_FOLDER)/hybrid_base.o: $(SRC_FOLDER)/hybrid_base.cu
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CCNV) -c $(SRC_FOLDER)/hybrid_base.cu -o $@ $(LIB_FLAGS_NV) $(INCLUDES_NV)

$(OBJ_FOLDER)/hybrid_unified.o: $(SRC_FOLDER)/hybrid_unified.cu
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CCNV) -c $(SRC_FOLDER)/hybrid_unified.cu -o $@ $(LIB_FLAGS_NV) $(INCLUDES_NV)

$(OBJ_FOLDER)/hybrid_pinned.o: $(SRC_FOLDER)/hybrid_pinned.cu
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CCNV) -c $(SRC_FOLDER)/hybrid_pinned.cu -o $@ $(LIB_FLAGS_NV) $(INCLUDES_NV)

$(OBJ_FOLDER)/cuda_no_copy.o: $(SRC_FOLDER)/cuda_no_copy.cu
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CCNV) -c $(SRC_FOLDER)/cuda_no_copy.cu -o $@ $(LIB_FLAGS_NV) $(INCLUDES_NV)

$(BIN_FOLDER)/$(MAIN_BIN): $(MAIN_SRC) $(OBJECTS)
	mkdir -p $(BIN_FOLDER)
	$(CCNV) $^ -o $@ $(LIBS) $(INCLUDES_NV) $(LIB_FLAGS_NV)

clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER)