CC=gcc

LIBS=
INCLUDES=-I../../ -Iinclude
LIB_FLAGS=-lm -Ofast -finline-functions -fopenmp


BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
BATCH_OUT_FOLDER := outputs


MAIN_NAME=main
MAIN_BIN=$(MAIN_NAME)
MAIN_SRC=$(MAIN_NAME).c

OBJECTS = $(OBJ_FOLDER)/my_time_lib.o $(OBJ_FOLDER)/cpu_sequential.o $(OBJ_FOLDER)/cpu_simd.o $(OBJ_FOLDER)/cpu_simd_parallel_dp.o

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

$(BIN_FOLDER)/$(MAIN_BIN): $(MAIN_SRC) $(OBJECTS)
	mkdir -p $(BIN_FOLDER)
	$(CC) $^ -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)

clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER)