PROJ_NAME = pathfinder
BUILD_TYPE = RELEASE
CUDA_ARCH = sm_50

# Directories
SRC_DIR = src
INC_DIR = inc
BUILD_DIR = build
OBJ_DIR = build/obj
BIN_DIR = build/bin

# Source files
SOURCES = $(SRC_DIR)/pathfinder.cu $(SRC_DIR)/main.cu $(SRC_DIR)/utility.cu $(SRC_DIR)/fieldgenerator.cu

# Object files
OBJECTS = $(OBJ_DIR)/pathfinder.obj $(OBJ_DIR)/main.obj $(OBJ_DIR)/utility.obj $(OBJ_DIR)/fieldgenerator.obj

# Compile-time define macros
ifeq ($(BUILD_TYPE), DEBUG)
	PP_DEFINE = DEBUG
else 
	ifeq ($(BUILD_TYPE), RELEASE)
		PP_DEFINE = NDEBUG
	endif
endif

# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -arch=$(CUDA_ARCH) -I $(INC_DIR) 
PPFLAGS = -D $(PP_DEFINE)

# Build object files
$(OBJ_DIR)/%.obj: $(SRC_DIR)/%.cu
	mkdir -p $(@D)
	$(NVCC) $< $(PPFLAGS) $(CFLAGS) -c -o $@

# Build executable
PHONY: build
build: $(OBJECTS)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(CFLAGS) $^ -o $(BIN_DIR)/$(PROJ_NAME)

# Clean
PHONY: clean
clean:
	rm -rf build/obj/

# Remove
remove:
	rm -rf build