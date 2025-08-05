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

# 默认配置
TARGET ?= ip
DEVICE ?= xcv80-lsva4737-2MHP-e-S
FREQ ?= 300

# 自动检测当前目录下的.cpp文件作为IP名
CPP_FILES := $(wildcard *.cpp)
IP_NAMES := $(basename $(CPP_FILES))

# 如果用户指定了IP名，使用指定的；否则使用检测到的所有IP
ifdef IP
	SELECTED_IPS := $(IP)
else
	SELECTED_IPS := $(IP_NAMES)
endif

# 为每个IP生成构建目录名
BUILD_DIRS := $(foreach ip,$(SELECTED_IPS),build_$(ip).$(DEVICE))

.PHONY: all clean help list $(IP_NAMES) rebuild

all: $(BUILD_DIRS)

# 通用规则：为每个IP构建
build_%.$(DEVICE): %.cpp
	@echo "Building IP: $(basename $(notdir $<)) from $<"
	@if [ ! -d "$@" ]; then \
		vitis_hls build.tcl -tclargs $(TARGET) $(DEVICE) $(basename $(notdir $<)) $(FREQ); \
	else \
		echo "Build directory $@ already exists, skipping..."; \
	fi

# 构建特定IP的快捷方式
$(foreach ip,$(IP_NAMES),$(eval $(ip): build_$(ip).$(DEVICE)))

# 清理所有构建目录
clean:
	rm -rf build_*.$(DEVICE) vitis_hls.log

# 强制重新构建
rebuild: clean all

# 构建特定IP（强制重新构建）
force-%: 
	rm -rf build_$*.$(DEVICE)
	$(MAKE) build_$*.$(DEVICE)

# 列出可用的IP
list:
	@echo "Available IP cores:"
	@for ip in $(IP_NAMES); do \
		echo "  - $$ip"; \
	done

# 帮助信息
help:
	@echo "HLS IP Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make                    # Build all IP cores found in current directory"
	@echo "  make <ip_name>          # Build specific IP core directly"
	@echo "  make IP=<name>          # Build specific IP core (alternative syntax)"
	@echo "  make DEVICE=<device>    # Set target device"
	@echo "  make FREQ=<freq>        # Set target frequency in MHz"
	@echo "  make TARGET=<target>    # Set build target (ip, syn, sim, cosim, all) - default: ip"
	@echo "  make clean              # Clean all build directories"
	@echo "  make rebuild            # Clean and rebuild all"
	@echo "  make force-<ip_name>    # Force rebuild specific IP"
	@echo "  make list               # List available IP cores"
	@echo ""
# 	@echo "Examples:"
# 	@echo "  make                           # Build all IPs as IP cores (default)"
# 	@echo "  make axpy                      # Build axpy IP core directly"
# 	@echo "  make IP=axpy                   # Build axpy IP core (alternative)"
# 	@echo "  make FREQ=400                  # Build all IPs at 400MHz"
# 	@echo "  make axpy TARGET=syn           # Build axpy for synthesis only"
# 	@echo "  make force-projlub             # Force rebuild projlub IP"
	@echo ""
	@echo "Current settings:"
	@echo "  TARGET: $(TARGET)"
	@echo "  DEVICE: $(DEVICE)"
	@echo "  FREQ:   $(FREQ) MHz"
	@echo "  Available IPs: $(IP_NAMES)"
