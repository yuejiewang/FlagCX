# Copyright (c) 2026 BAAI. All rights reserved.
# Modified by 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved
# Modified by 2025 DU. All Rights Reserved.
BUILDDIR ?= $(abspath ./build)

# set to 0 if not provided
USE_NVIDIA ?= 0
USE_ASCEND ?= 0
USE_ILUVATAR_COREX ?= 0
USE_CAMBRICON ?= 0
USE_GLOO ?= 0
USE_BOOTSTRAP ?= 0
USE_METAX ?= 0
USE_MUSA ?= 0
USE_KUNLUNXIN ?=0
USE_AMD ?= 0
USE_DU ?= 0
USE_TSM ?= 0
USE_MPI ?= 0
USE_UCX ?= 0
USE_IBUC ?= 0
USE_ENFLAME ?= 0
USE_SUNRISE ?= 0
COMPILE_KERNEL ?= 0
ASCEND_SOC_VERSION ?= $(SOC_VERSION)
ASCEND_DRIVER_HOME ?= /usr/local/Ascend/driver
CMAKE ?= cmake

# set to empty if not provided
DEVICE_HOME ?=
CCL_HOME ?=
HOST_CCL_HOME ?=
MPI_HOME ?=
UCX_HOME ?=

ifeq ($(strip $(DEVICE_HOME)),)
	ifeq ($(USE_NVIDIA), 1)
		DEVICE_HOME = /usr/local/cuda
	else ifeq ($(USE_ASCEND), 1)
		DEVICE_HOME = $(if $(strip $(ASCEND_HOME_PATH)),$(ASCEND_HOME_PATH),/usr/local/Ascend/cann)
	else ifeq ($(USE_ILUVATAR_COREX), 1)
		DEVICE_HOME = /usr/local/corex
	else ifeq ($(USE_CAMBRICON), 1)
		DEVICE_HOME = $(NEUWARE_HOME)
	else ifeq ($(USE_METAX), 1)
		DEVICE_HOME = /opt/maca
	else ifeq ($(USE_MUSA), 1)
		DEVICE_HOME = /usr/local/musa
	else ifeq ($(USE_KUNLUNXIN), 1)
		DEVICE_HOME = /usr/local/xpu
	else ifeq ($(USE_DU), 1)
		DEVICE_HOME = ${CUDA_PATH}
	else ifeq ($(USE_AMD), 1)
		DEVICE_HOME = /opt/rocm
	else ifeq ($(USE_TSM), 1)
		DEVICE_HOME = /usr/local/kuiper
	else ifeq ($(USE_ENFLAME), 1)
		DEVICE_HOME = /opt/tops/
	else ifeq ($(USE_SUNRISE), 1)
		DEVICE_HOME = /usr/local/tangrt
	else
		DEVICE_HOME = /usr/local/cuda
	endif
endif

ifeq ($(strip $(CCL_HOME)),)
	ifeq ($(USE_NVIDIA), 1)
		CCL_HOME = /usr/local/nccl/build
	else ifeq ($(USE_ASCEND), 1)
		CCL_HOME = $(DEVICE_HOME)
	else ifeq ($(USE_ILUVATAR_COREX), 1)
		CCL_HOME = /usr/local/corex
	else ifeq ($(USE_CAMBRICON), 1)
		CCL_HOME = $(NEUWARE_HOME)
	else ifeq ($(USE_METAX), 1)
		CCL_HOME = /opt/maca
	else ifeq ($(USE_MUSA), 1)
		CCL_HOME = /usr/local/musa
	else ifeq ($(USE_KUNLUNXIN), 1)
		CCL_HOME = /usr/local/xccl
	else ifeq ($(USE_DU), 1)
		CCL_HOME = ${CUDA_PATH}
	else ifeq ($(USE_AMD), 1)
		CCL_HOME = /opt/rocm
	else ifeq ($(USE_TSM), 1)
		CCL_HOME = /usr/local/kuiper
	else ifeq ($(USE_ENFLAME), 1)
		CCL_HOME = /usr
	else ifeq ($(USE_SUNRISE), 1)
		CCL_HOME = /usr/local/pccl
	else
		CCL_HOME = /usr/local/nccl/build
	endif
endif

ifeq ($(strip $(HOST_CCL_HOME)),)
	ifeq ($(USE_GLOO), 1)
		HOST_CCL_HOME = /usr/local
	else ifeq ($(USE_MPI), 1)
		HOST_CCL_HOME = $(MPI_HOME)
	else
		HOST_CCL_HOME = 
	endif
endif

ifeq ($(strip $(MPI_HOME)),)
	ifeq ($(USE_MPI), 1)
		MPI_HOME = /usr/local
	endif
endif

ifeq ($(strip $(UCX_HOME)),)
	ifeq ($(USE_UCX), 1)
		UCX_HOME = /usr/local/ucx
	endif
endif

DEVICE_LIB =
DEVICE_INCLUDE =
DEVICE_LINK =
DEVICE_PLATFORM =
DEVICE_COMPILER =
DEVICE_COMPILE_FLAG =
DEVICE_LINK_FLAG =
DEVICE_FILE_EXTENSION =
ASCEND_KERNEL_RPATH =
ASCEND_KERNEL_CLEAN =
CCL_LIB =
CCL_INCLUDE =
CCL_LINK =
HOST_CCL_LIB = 
HOST_CCL_INCLUDE =
HOST_CCL_LINK =
ADAPTOR_FLAG =
HOST_CCL_ADAPTOR_FLAG =
UCX_LIB =
UCX_INCLUDE =
UCX_LINK =
NET_ADAPTOR_FLAG =
COMPILE_KERNEL_HOST_FLAG=
COMPILE_KERNEL_FLAG =
HOST_COMPILER ?= g++
ifeq ($(USE_NVIDIA), 1)
	include makefiles/nvidia_gencode.mk
	DEVICE_LIB = $(DEVICE_HOME)/lib64
	DEVICE_INCLUDE = $(DEVICE_HOME)/include $(DEVICE_HOME)/include/cccl
	DEVICE_LINK = -lcudart -lcuda
	DEVICE_PLATFORM = CUDA
	DEVICE_COMPILER = $(DEVICE_HOME)/bin/nvcc
	DEVICE_COMPILE_FLAG = -c --cudart=shared -Xcompiler -fPIC -MMD -MP -rdc=true -g $(DEVICE_COMPILER_GENCODE)
	DEVICE_LINK_FLAG = --cudart=shared -Xcompiler -fPIC $(DEVICE_COMPILER_GENCODE)
	DEVICE_FILE_EXTENSION = cu
	CCL_LIB = $(CCL_HOME)/lib
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lnccl
	ADAPTOR_FLAG = -DUSE_NVIDIA_ADAPTOR
ifeq ($(NVCC_GENCODE_MULTICAST_UNSUPPORTED), 1)
	ADAPTOR_FLAG += -DNVCC_GENCODE_MULTICAST_UNSUPPORTED
endif
else ifeq ($(USE_ASCEND), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib64
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	# CANN 9.x publishes the acl/acl_rt.h runtime ABI from libacl_rt.so.
	DEVICE_LINK = -lacl_rt \
		-L$(ASCEND_DRIVER_HOME)/lib64/driver \
		-Wl,-rpath,$(ASCEND_DRIVER_HOME)/lib64/driver \
		-lascend_hal
	CCL_LIB = $(CCL_HOME)/lib64
	CCL_INCLUDE = $(CCL_HOME)/include
	# UniRunner's HCCS AlltoAll path uses the communication-operator
	# primitives exported by libhcomm in addition to the HCCL control plane.
	CCL_LINK = -lhccl -lhcomm -lc_sec
	ADAPTOR_FLAG = -DUSE_ASCEND_ADAPTOR
ifeq ($(COMPILE_KERNEL), 1)
	# The CMake-built launcher library stays next to its target-specific
	# build tree during development and is installed next to libflagcx.so.
	ASCEND_KERNEL_RPATH = \
		-Wl,-rpath,$(ASCEND_KERNEL_BUILD_DIR)/lib \
		-Wl,-rpath,'$$ORIGIN'
	ASCEND_KERNEL_CLEAN = \
		$(DESTDIR)/libflagcx_ascend_unirunner.so
endif
else ifeq ($(USE_ILUVATAR_COREX), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lcudart -lcuda
	CCL_LIB = $(CCL_HOME)/lib
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lnccl
	ADAPTOR_FLAG = -DUSE_ILUVATAR_COREX_ADAPTOR
else ifeq ($(USE_CAMBRICON), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib64
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lcnrt
	CCL_LIB = $(CCL_HOME)/lib64
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lcncl
	ADAPTOR_FLAG = -DUSE_CAMBRICON_ADAPTOR
else ifeq ($(USE_METAX), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib64
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	CCL_LIB = $(CCL_HOME)/lib64
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lmccl
	ADAPTOR_FLAG = -DUSE_METAX_ADAPTOR
else ifeq ($(USE_MUSA), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	CCL_LIB = $(CCL_HOME)/lib
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lmccl -lmusa
	ADAPTOR_FLAG = -DUSE_MUSA_ADAPTOR
else ifeq ($(USE_KUNLUNXIN), 1)
	DEVICE_LIB = $(DEVICE_HOME)/so
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lxpurt -lcudart
	CCL_LIB = $(CCL_HOME)/so
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lbkcl
	ADAPTOR_FLAG = -DUSE_KUNLUNXIN_ADAPTOR
else ifeq ($(USE_DU), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib64
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lcudart -lcuda
	CCL_LIB = $(CCL_HOME)/lib64
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lnccl
	ADAPTOR_FLAG = -DUSE_DU_ADAPTOR
	DEVICE_PLATFORM = DU
	DEVICE_COMPILER = $(DEVICE_HOME)/bin/nvcc
	DEVICE_COMPILE_FLAG = -c --cudart=shared -Xcompiler -fPIC -MMD -MP -rdc=true -g
	DEVICE_LINK_FLAG = --cudart=shared -Xcompiler -fPIC
	DEVICE_FILE_EXTENSION = cu
else ifeq ($(USE_AMD), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lhiprtc
	CCL_LIB = $(CCL_HOME)/lib
	CCL_INCLUDE = $(CCL_HOME)/include/rccl
	CCL_LINK = -lrccl
	ADAPTOR_FLAG = -DUSE_AMD_ADAPTOR -D__HIP_PLATFORM_AMD__
else ifeq ($(USE_TSM), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -lhpgr
	CCL_LIB = $(CCL_HOME)/lib
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -ltccl
	ADAPTOR_FLAG = -DUSE_TSM_ADAPTOR
else ifeq ($(USE_ENFLAME), 1)
	DEVICE_LIB = $(DEVICE_HOME)/lib
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -ltopsrt
	CCL_LIB = $(CCL_HOME)/lib
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -leccl
	ADAPTOR_FLAG = -DUSE_ENFLAME_ADAPTOR
else ifeq ($(USE_SUNRISE), 1)
	DEVICE_LIB = $(DEVICE_HOME)/targets/linux-x86_64/lib
	DEVICE_INCLUDE = $(DEVICE_HOME)/include
	DEVICE_LINK = -ltangrt_shared
	CCL_LIB = $(CCL_HOME)/lib/linux-x86_64
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lpccl
	ADAPTOR_FLAG = -DUSE_SUNRISE_ADAPTOR
else
	DEVICE_LIB = $(DEVICE_HOME)/lib64
	DEVICE_INCLUDE = $(DEVICE_HOME)/include $(DEVICE_HOME)/include/cccl
	DEVICE_LINK = -lcudart -lcuda
	DEVICE_PLATFORM = CUDA
	DEVICE_COMPILER = $(DEVICE_HOME)/bin/nvcc
	DEVICE_COMPILE_FLAG = -c --cudart=shared -Xcompiler -fPIC -MMD -MP -rdc=true -g $(DEVICE_COMPILER_GENCODE)
	DEVICE_LINK_FLAG = --cudart=shared -Xcompiler -fPIC $(DEVICE_COMPILER_GENCODE)
	DEVICE_FILE_EXTENSION = cu
	CCL_LIB = $(CCL_HOME)/lib
	CCL_INCLUDE = $(CCL_HOME)/include
	CCL_LINK = -lnccl
	ADAPTOR_FLAG = -DUSE_NVIDIA_ADAPTOR
ifeq ($(NVCC_GENCODE_MULTICAST_UNSUPPORTED), 1)
	ADAPTOR_FLAG += -DNVCC_GENCODE_MULTICAST_UNSUPPORTED
endif
	USE_NVIDIA := 1
endif

ifeq ($(FORCE_DEFAULT_PATH), 1)
	ADAPTOR_FLAG += -DFLAGCX_FORCE_DEFAULT_PATH
endif

ifeq ($(USE_GLOO), 1)
	HOST_CCL_LIB = $(HOST_CCL_HOME)/lib
	HOST_CCL_INCLUDE = $(HOST_CCL_HOME)/include
	HOST_CCL_LINK = -lgloo -libverbs
	HOST_CCL_ADAPTOR_FLAG = -DUSE_GLOO_ADAPTOR
else ifeq ($(USE_MPI), 1)
	HOST_CCL_LIB = $(MPI_HOME)/lib
	HOST_CCL_INCLUDE = $(MPI_HOME)/include
	HOST_CCL_LINK = -lmpi
	HOST_CCL_ADAPTOR_FLAG = -DUSE_MPI_ADAPTOR
else ifeq ($(USE_BOOTSTRAP), 1)
	HOST_CCL_LIB = /usr/local/lib
	HOST_CCL_INCLUDE = /usr/local/include
	HOST_CCL_LINK = 
	HOST_CCL_ADAPTOR_FLAG = -DUSE_BOOTSTRAP_ADAPTOR
else
	HOST_CCL_LIB = /usr/local/lib
	HOST_CCL_INCLUDE = /usr/local/include
	HOST_CCL_LINK = 
	HOST_CCL_ADAPTOR_FLAG = -DUSE_BOOTSTRAP_ADAPTOR
endif

# UCX network adaptor configuration
ifeq ($(USE_UCX), 1)
	UCX_LIB = $(UCX_HOME)/lib
	UCX_INCLUDE = $(UCX_HOME)/include
	UCX_LINK = -lucp -lucs -luct
	NET_ADAPTOR_FLAG = -DUSE_UCX
else
	UCX_LIB = $(UCX_HOME)/lib
	UCX_INCLUDE = $(UCX_HOME)/include
	UCX_LINK = 
	NET_ADAPTOR_FLAG = 
endif

# IBUC network adaptor configuration
ifeq ($(USE_IBUC), 1)
	NET_ADAPTOR_FLAG += -DUSE_IBUC
endif

ifeq ($(COMPILE_KERNEL), 1)
	COMPILE_KERNEL_FLAG = -DCOMPILE_KERNEL
	COMPILE_KERNEL_HOST_FLAG = -DCOMPILE_KERNEL_HOST
endif

LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj
BUILD_INCDIR := $(BUILDDIR)/include
ASCEND_KERNEL_SRC_DIR := $(abspath flagcx/kernels/ascend)
ASCEND_KERNEL_BUILD_DIR := \
	$(OBJDIR)/ascend-unirunner/$(ASCEND_SOC_VERSION)
ASCEND_KERNEL_LIB := \
	$(ASCEND_KERNEL_BUILD_DIR)/lib/libflagcx_ascend_unirunner.so
ASCEND_KERNEL_SOURCES := \
	$(ASCEND_KERNEL_SRC_DIR)/CMakeLists.txt \
	$(ASCEND_KERNEL_SRC_DIR)/flagcx_ascend_unirunner_reduce.cpp
PREFIX ?= /usr/local
DESTDIR  ?= $(PREFIX)/lib
INC_DESTDIR ?= $(PREFIX)/include

# Public headers exported alongside libflagcx.so
PUBLIC_HEADERS := \
	flagcx/include/flagcx.h \
	flagcx/include/flagcx_kernel.h \
	flagcx/include/flagcx_p2p.h
BUILD_PUBLIC_HEADERS := $(PUBLIC_HEADERS:flagcx/include/%=$(BUILD_INCDIR)/%)

INCLUDEDIR := \
	$(abspath flagcx/include) \
	$(abspath flagcx/adaptor/include) \
	$(abspath flagcx/runner/include) \
	$(abspath flagcx/core/include) \
	$(abspath flagcx/service/include) \
	$(abspath third-party/json/single_include)

LIBSRCFILES:= \
	$(wildcard flagcx/*.cc) \
	$(wildcard flagcx/adaptor/*.cc) \
	$(wildcard flagcx/adaptor/device/*.cc) \
	$(wildcard flagcx/adaptor/ccl/*.cc) \
	$(wildcard flagcx/adaptor/net/*.cc) \
	$(wildcard flagcx/adaptor/tuner/*.cc) \
	$(wildcard flagcx/runner/*.cc) \
	$(wildcard flagcx/core/*.cc) \
	$(wildcard flagcx/service/*.cc)

ifeq ($(COMPILE_KERNEL), 1)
ifeq ($(USE_ASCEND), 1)
DEVSRCFILES :=
DEVOBJ :=
else
DEVSRCFILES:= \
	$(wildcard flagcx/kernels/*.$(DEVICE_FILE_EXTENSION))
ifneq ($(USE_NVIDIA), 1)
EXCLUDE_SOURCES := custom_allreduce.cu
else
EXCLUDE_SOURCES :=
endif
DEVSRCFILES := $(filter-out flagcx/kernels/$(EXCLUDE_SOURCES), $(DEVSRCFILES))
DEVOBJ:= $(DEVSRCFILES:%.$(DEVICE_FILE_EXTENSION)=$(OBJDIR)/%.o)
endif
endif
LIBOBJ:= $(LIBSRCFILES:%.cc=$(OBJDIR)/%.o)

TARGET = libflagcx.so
all: $(LIBDIR)/$(TARGET) $(BUILD_PUBLIC_HEADERS)

print_var:
	@echo "USE_KUNLUNXIN : $(USE_KUNLUNXIN)"
	@echo "DEVICE_HOME: $(DEVICE_HOME)"
	@echo "CCL_HOME: $(CCL_HOME)"
	@echo "HOST_CCL_HOME: $(HOST_CCL_HOME)"
	@echo "MPI_HOME: $(MPI_HOME)"
	@echo "USE_NVIDIA: $(USE_NVIDIA)"
	@echo "USE_ASCEND: $(USE_ASCEND)"
	@echo "USE_ILUVATAR_COREX: $(USE_ILUVATAR_COREX)"
	@echo "USE_CAMBRICON: $(USE_CAMBRICON)"
	@echo "USE_KUNLUNXIN: $(USE_KUNLUNXIN)"
	@echo "USE_GLOO: $(USE_GLOO)"
	@echo "USE_MPI: $(USE_MPI)"
	@echo "USE_MUSA: $(USE_MUSA)"
	@echo "USE_DU: $(USE_DU)"
	@echo "USE_AMD: $(USE_AMD)"
	@echo "USE_TSM: $(USE_TSM)"
	@echo "USE_ENFLAME: $(USE_ENFLAME)"
	@echo "COMPILE_KERNEL: $(COMPILE_KERNEL)"
	@echo "ASCEND_SOC_VERSION: $(ASCEND_SOC_VERSION)"
	@echo "DEVICE_LIB: $(DEVICE_LIB)"
	@echo "DEVICE_INCLUDE: $(DEVICE_INCLUDE)"
	@echo "CCL_LIB: $(CCL_LIB)"
	@echo "CCL_INCLUDE: $(CCL_INCLUDE)"
	@echo "HOST_CCL_LIB: $(HOST_CCL_LIB)"
	@echo "HOST_CCL_INCLUDE: $(HOST_CCL_INCLUDE)"
	@echo "ADAPTOR_FLAG: $(ADAPTOR_FLAG)"
	@echo "HOST_CCL_ADAPTOR_FLAG: $(HOST_CCL_ADAPTOR_FLAG)"
	@echo "USE_UCX: $(USE_UCX)"
	@echo "UCX_HOME: $(UCX_HOME)"
	@echo "UCX_LIB: $(UCX_LIB)"
	@echo "UCX_INCLUDE: $(UCX_INCLUDE)"
	@echo "USE_IBUC: $(USE_IBUC)"
	@echo "NET_ADAPTOR_FLAG: $(NET_ADAPTOR_FLAG)"
	@echo "DEVSRCFILES: $(DEVSRCFILES)"

ifeq ($(COMPILE_KERNEL), 1)
ifeq ($(USE_ASCEND), 1)
DEVOBJS = $(ASCEND_KERNEL_LIB)
else
DEVOBJS = $(DEVOBJ) $(OBJDIR)/kernel_dlink.o
endif
else
DEVOBJS =
endif

HOST_LINKER   ?= $(HOST_COMPILER)
DEVICE_LINKER := $(DEVICE_COMPILER)
LINKER        := $(HOST_LINKER)
ifeq ($(COMPILE_KERNEL)$(USE_DU),11)
  LINKER := $(DEVICE_LINKER)
endif

$(LIBDIR)/$(TARGET): $(LIBOBJ) $(DEVOBJS)
	@mkdir -p `dirname $@`
	@echo "Linking   $@"
	@$(LINKER) $^ -o $@ -L$(CCL_LIB) -L$(DEVICE_LIB) -L$(HOST_CCL_LIB) -L$(UCX_LIB) -shared -fvisibility=default -Wl,--no-as-needed -Wl,-rpath,$(LIBDIR) -Wl,-rpath,$(CCL_LIB) -Wl,-rpath,$(HOST_CCL_LIB) -Wl,-rpath,$(UCX_LIB) $(ASCEND_KERNEL_RPATH) -lpthread -lrt -ldl $(CCL_LINK) $(DEVICE_LINK) $(HOST_CCL_LINK) $(UCX_LINK) -g

# Copy public headers from flagcx/include/ into the build output tree so they
# sit next to the shared libraries (build/include + build/lib).
$(BUILD_INCDIR)/%.h: flagcx/include/%.h
	@mkdir -p `dirname $@`
	@echo "Copying   $@"
	@cp $< $@

$(OBJDIR)/%.o: %.cc
	@mkdir -p `dirname $@`
	@echo "Compiling $@"
	@$(HOST_COMPILER) $< -o $@ $(foreach dir,$(INCLUDEDIR),-I$(dir)) -I$(CCL_INCLUDE) $(addprefix -I,$(DEVICE_INCLUDE)) -I$(HOST_CCL_INCLUDE) -I$(UCX_INCLUDE) $(ADAPTOR_FLAG) $(HOST_CCL_ADAPTOR_FLAG) $(NET_ADAPTOR_FLAG) $(COMPILE_KERNEL_HOST_FLAG) -c -fPIC -fvisibility=default -Wvla -Wno-unused-function -Wno-sign-compare -Wall -MMD -MP -g

ifeq ($(COMPILE_KERNEL), 1)
ifeq ($(USE_ASCEND), 1)
$(ASCEND_KERNEL_LIB): $(ASCEND_KERNEL_SOURCES)
	@echo "Compiling $@ (ASCEND)"
	@$(CMAKE) -S "$(ASCEND_KERNEL_SRC_DIR)" \
		-B "$(ASCEND_KERNEL_BUILD_DIR)" \
		-DASCEND_CANN_PACKAGE_PATH="$(DEVICE_HOME)" \
		-DSOC_VERSION="$(ASCEND_SOC_VERSION)" \
		-DRUN_MODE=npu \
		-DCMAKE_BUILD_TYPE=Release
	@$(CMAKE) --build "$(ASCEND_KERNEL_BUILD_DIR)" \
		--target flagcx_ascend_unirunner
else
$(OBJDIR)/kernel_dlink.o: $(DEVOBJ)
	@$(DEVICE_LINKER) -dlink $^ -o $@ $(DEVICE_LINK) $(DEVICE_LINK_FLAG)

$(OBJDIR)/%.o: %.$(DEVICE_FILE_EXTENSION)
	@mkdir -p `dirname $@`
	@echo "Compiling $@ ($(DEVICE_PLATFORM))"
	@$(DEVICE_COMPILER) $< -o $@ $(foreach dir,$(INCLUDEDIR),-I$(dir)) -I$(CCL_INCLUDE) $(addprefix -I,$(DEVICE_INCLUDE)) -I$(HOST_CCL_INCLUDE) -I$(UCX_INCLUDE) $(ADAPTOR_FLAG) $(HOST_CCL_ADAPTOR_FLAG) $(NET_ADAPTOR_FLAG) $(DEVICE_COMPILE_FLAG) $(COMPILE_KERNEL_FLAG) -g
endif
endif

ifeq ($(COMPILE_KERNEL), 1)
-include $(LIBOBJ:.o=.d) $(DEVOBJ:.o=.d)
else
-include $(LIBOBJ:.o=.d)
endif

INSTALLDIR := /usr/local/lib
install:
	@mkdir -p $(DESTDIR)
	@cp $(LIBDIR)/$(TARGET) $(DESTDIR)/$(TARGET)
ifeq ($(COMPILE_KERNEL)$(USE_ASCEND),11)
	@cp $(ASCEND_KERNEL_LIB) $(DESTDIR)/
endif
	@mkdir -p $(INC_DESTDIR)
	@cp $(PUBLIC_HEADERS) $(INC_DESTDIR)/

clean:
	@rm -rf $(LIBDIR)/$(TARGET) $(DESTDIR)/$(TARGET) \
		$(ASCEND_KERNEL_CLEAN) $(BUILD_INCDIR) $(OBJDIR)
