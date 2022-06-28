###########################################################

##### USER SPECIFIC DIRECTORIES #####
##### ABSOLUTE PATH TO CUDA DIRECTORY #####
##### MCMPC toolkit supports CUDA 9.0 or later #####

# CUDA_ROOT_DIR=/usr/local/cuda-9.0
CUDA_ROOT_DIR=/usr/local/cuda-10.2
# CUDA_ROOT_DIR=/usr/local/cuda-11.2

###########################################################

##### CC COMPILER OPTIONS #####
## C++ compiler options: ##
CC=g++
CC_FLAG=
CC_LIBS =

###########################################################

##### NVCC COMPILER OPTIONS #####
## NVCC compiler options: ##
NVCC=nvcc
NVCC_FLAGS=
NVCC_LIBS=

## Path to CUDA library directory: ##
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64

## Path to CUDA include directory: ##
CUDA_INC_DIR =-I$(CUDA_ROOT_DIR)/include

## Path to CUDA linking libraries ##
CUDA_LINK_LIBS = -lcudart -lcusolver -lcublas

###########################################################
##### FILE EXTENSION #####
## file extension of source file ##
SRCEXT=.cu

## file extension of object file ##
OBJEXT=.o


###########################################################

##### PROJECT STRUCTURE  #####
## Path from home directory to Current directory ##
CURRENT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

## Path to MCMPC toolkit object files ##
# MCMPC_OBJS_DIR = ../../bin
PARENT_DIR := $(shell dirname $(CURRENT_DIR))
MCMPC_MAIN_DIR := $(shell dirname $(PARENT_DIR))
MCMPC_OBJS_DIR := $(MCMPC_MAIN_DIR)/bin

## Path to MCMPC toolkit source files ##
MCMPC_SRC_DIR := $(MCMPC_MAIN_DIR)/src

## Object file directory ##
OBJ_DIR = ./bin

## Data output directory ##
DATA_DIR = ./output

###########################################################

## Make variables ##

## Target executable file name ##
EXE = exe_mcmpc

## Materials in mcmpc-code/src directory ##
MCMPC_SOURCES := $(shell find $(MCMPC_SRC_DIR)/*$(SRCEXT))

## MAterials in current directory ##
SRCS := $(shell find $(CURRENT_DIR)/*$(SRCEXT))

## Object files ##
MCMPC_OBJS := $(patsubst $(MCMPC_SRC_DIR)/%,$(MCMPC_OBJS_DIR)/%,$(subst $(SRCEXT),$(OBJEXT), $(MCMPC_SOURCES)))
OTHER_OBJS := $(patsubst $(CURRENT_DIR)/%, $(OBJ_DIR)/%, $(subst $(SRCEXT),$(OBJEXT), $(SRCS)))

###########################################################

## command ##
MKDIR = mkdir

###########################################################

## method ##
all: $(DATA_DIR) $(MCMPC_OBJS_DIR) $(OBJ_DIR) $(MCMPC_OBJS) $(OTHER_OBJS) $(EXE) 

$(DATA_DIR) :
	@if[! -d $(DATA_DIR) ]; then \
	echo "$(MKDIR) $(DATA_DIR)";  $(MKDIR) $(DATA_DIR); \
	fi

$(MCMPC_OBJS_DIR) : $(wildcard *.o)
	@if[! -d $(MCMPC_OBJS_DIR) ]; then \
	echo "$(MKDIR) $(MCMPC_OBJS_DIR)";  $(MKDIR) $(MCMPC_OBJS_DIR);\
	fi

$(OBJ_DIR) : $(wildcard *.o)
	@if[! -d $(OBJ_DIR)]; then \
	echo "$(MKDIR) $(OBJ_DIR)"; $(MKDIR) $(OBJ_DIR); \
	fi

$(EXE) : $(MCMPC_OBJS) $(OTHER_OBJS)
	@echo "Building executable files $(EXE) ..." 
	@$(NVCC) $(NVCC_FLAGS) $^  -o $@ $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)
	@echo "Success building executable files $(EXE) ..."

$(OBJ_DIR)/%.o : %.cu
	@echo "Building $@ ..."
	@$(NVCC) $(NVCC_FLAGS) -dc $< -o $@ $(CUDA_INC_DIR) $(CUDA_LINK_LIBS)
	@echo "Success Building $@ ..."

$(MCMPC_OBJS_DIR)/%.o : $(MCMPC_SRC_DIR)/%.cu
	@echo "Building $@ ..."
	@$(NVCC) $(NVCC_FLAGS) -dc $< -o $@ $(CUDA_INC_DIR)
	@echo "Success Building $@ ..."
clean:
	@$(RM) $(EXE)
	@$(RM) -rf $(OBJ_DIR) 





