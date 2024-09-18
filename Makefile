###############################################################################
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################
#
# CUDA Samples
#
###############################################################################





TARGET_ARCH ?= $(shell uname -m) # 定义目标体系结构

PROJECTS ?= $(shell find Samples -name Makefile)  # 通过 find 命令在 Samples 目录中找到所有包含 Makefile 的子项目。每个子项目都视作一个独立的 CUDA 示例项目

FILTER_OUT := # 预定义为空，用于后续过滤项目，可以手动设定不需要构建的项目

PROJECTS := $(filter-out $(FILTER_OUT),$(PROJECTS)) # 根据 FILTER_OUT 变量过滤出需要构建的项目

# 为每个匹配的目标执行构建操作，所有以 .ph_build 结尾的目标
%.ph_build :
# 再次调用 Make 以执行子目录中的 Makefile ,-C 选项告诉 Make 切换到指定目录进行构建。$(dir $*) 取的是目标文件路径部分，其中 $* 是通配符 % 匹配到的部分，$(dir $*) 提取的是该路径的目录部分
	+@$(MAKE) -C $(dir $*) $(MAKECMDGOALS)   

%.ph_test :
	+@$(MAKE) -C $(dir $*) testrun

%.ph_clean : 
	+@$(MAKE) -C $(dir $*) clean $(USE_DEVICE)

%.ph_clobber :
	+@$(MAKE) -C $(dir $*) clobber $(USE_DEVICE)

all:  $(addsuffix .ph_build,$(PROJECTS)) # 默认的目标，构建所有项目
	@echo "Finished building CUDA samples"

build: $(addsuffix .ph_build,$(PROJECTS)) # build：显式构建目标，与 all 类似，但只进行构建，不输出成功信息

test : $(addsuffix .ph_test,$(PROJECTS)) # test：构建测试目标，与 all 类似，但只进行测试，不输出成功信息

tidy:									# 清理临时文件，例如以 # 或 ~ 结尾的文件，通常是编辑器生成的备份文件
	@find * | egrep "#" | xargs rm -f
	@find * | egrep "\~" | xargs rm -f

clean: tidy $(addsuffix .ph_clean,$(PROJECTS))  # 执行 tidy 并清理所有项目中的生成文件

clobber: clean $(addsuffix .ph_clobber,$(PROJECTS)) # 执行更彻底的清理，删除所有生成的文件
