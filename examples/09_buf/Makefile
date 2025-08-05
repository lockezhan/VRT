# ##################################################################################################
#  The MIT License (MIT)
#  Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software
#  and associated documentation files (the "Software"), to deal in the Software without restriction,
#  including without limitation the rights to use, copy, modify, merge, publish, distribute,
#  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in all copies or
#  substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ##################################################################################################

# 默认设备配置
DEVICE ?= xcv80-lsva4737-2MHP-e-S

# 自动检测hls目录下的.cpp文件并生成对应的build目录
CPP_FILES := $(wildcard hls/*.cpp)
IP_NAMES := $(basename $(notdir $(CPP_FILES)))

DESIGN_NAME=09_buf
HOME_DIR=$(shell realpath .)
BUILD_DIR=$(shell realpath ./build)
HLS_DIR=$(shell realpath ./hls)
V80PP_PATH=$(shell realpath ../../submodules/v80-vitis-flow)
VPP_DIR=$(BUILD_DIR)/v80-vitis-flow

# 生成HLS构建目录路径（使用相对于HLS_DIR的路径）
HLS_BUILD_DIRS := $(foreach ip,$(IP_NAMES),$(HLS_DIR)/build_$(ip).$(DEVICE))

.PHONY: all setup hls hw emu sim app clean help

all: setup hls hw app

emu_all: setup hls emu app

sim_all: setup hls sim app

hw_all: setup hls hw app

setup:
	mkdir -p $(BUILD_DIR)
	cp -r $(V80PP_PATH) $(BUILD_DIR)

hls:
	@echo "Running HLS step"
	$(MAKE) -C $(HLS_DIR)

hw: setup hls
	@echo "Running HW step"
	@echo "Detected IP cores: $(IP_NAMES)"
	@echo "Using build directories: $(HLS_BUILD_DIRS)"
	cd $(VPP_DIR) && \
	./scripts/v80++ --design-name $(DESIGN_NAME) --cfg $(HOME_DIR)/config.cfg --platform hw --kernels $(foreach dir,$(HLS_BUILD_DIRS),$(dir)/sol1) && \
	cp build/$(DESIGN_NAME)_hw.vrtbin $(BUILD_DIR)

emu: setup hls
	@echo "Running EMU step"
	@echo "Detected IP cores: $(IP_NAMES)"
	@echo "Using build directories: $(HLS_BUILD_DIRS)"
	cd $(VPP_DIR) && \
	./scripts/v80++ --design-name $(DESIGN_NAME) --cfg $(HOME_DIR)/config.cfg --platform emu --segmented --kernels $(foreach dir,$(HLS_BUILD_DIRS),$(dir)/sol1) && \
	cp build/$(DESIGN_NAME)_emu.vrtbin $(BUILD_DIR)

sim: setup hls
	@echo "Running SIM step"
	@echo "Detected IP cores: $(IP_NAMES)"
	@echo "Using build directories: $(HLS_BUILD_DIRS)"
	cd $(VPP_DIR) && \
	./scripts/v80++ --design-name $(DESIGN_NAME) --cfg $(HOME_DIR)/config.cfg --platform sim --segmented --kernels $(foreach dir,$(HLS_BUILD_DIRS),$(dir)/sol1) && \
	cp build/$(DESIGN_NAME)_sim.vrtbin $(BUILD_DIR)

app: setup
	@echo "Running user app build step"
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
	cmake .. && \
	make -j9
	@echo "Setting LD_LIBRARY_PATH"
	export LD_LIBRARY_PATH=$$(dirname $$(which vivado))/../lib/lnx64.o:$$LD_LIBRARY_PATH
	@echo "Setting PATH"
	export PATH=$$PATH:/usr/local/sbin

clean:
	rm -rf $(BUILD_DIR)
	make -C $(HLS_DIR) clean


help:
	@echo "=========================="
	@echo ""
	@echo "Detected IP cores: $(IP_NAMES)"
# 	@echo "Target device: $(DEVICE)"
# 	@echo "HLS build directories: $(HLS_BUILD_DIRS)"
	@echo ""
	@echo "Usage:"
	@echo "  make [target] [DEVICE=device_name]"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build everything (setup + hls + hw + app)"
	@echo "  hw_all   - Build for hardware (setup + hls + hw + app)"
	@echo ""
	@echo "Examples:"
	@echo "  make                                   # Default build"
	@echo "  make DEVICE=xcv80-lsva4737-2MHP-e-L    # Use different device"