# omarchy-crt lab: CUDA virtual phosphor surface. Standalone; installs nothing.
TC      := .toolchain/nvidia
CXX     := g++
CXXFLAGS:= -O2 -std=c++17 -Wall -Wno-deprecated-declarations -I. -Isrc \
           -I$(TC)/cuda_runtime/include -I$(TC)/cuda_nvrtc/include
NVRTC   := $(abspath $(TC)/cuda_nvrtc/lib)
LDCUDA  := -lcuda -L$(NVRTC) -l:libnvrtc.so.12 -Wl,-rpath,$(NVRTC)
LDGL    := -lepoxy -lEGL
BINS    := bin/probe bin/stage1_test bin/stage2_bench bin/crt_render

all: $(BINS)

# The CUDA compiler and headers, into this folder only (about 320 MB, ignored by git).
toolchain:
	uv pip install -q --python-platform x86_64-manylinux_2_28 --python-version 3.12 \
	    --target .toolchain nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12

bin/probe: src/probe.cpp src/glrt.hpp src/cudart.hpp phosphor_params.h
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDCUDA) $(LDGL)
bin/stage1_test: src/stage1_test.cpp src/cudart.hpp src/cpu_ref.hpp phosphor_params.h
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDCUDA)
bin/stage2_bench: src/stage2_bench.cpp src/pipelines.hpp src/look.hpp src/cuda_path.hpp src/glrt.hpp src/cudart.hpp phosphor_params.h phosphor_half.h kernels/phosphor.cu
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDCUDA) $(LDGL)
bin/crt_render: src/crt_render.cpp kernels/glass.cu src/pipelines.hpp src/look.hpp src/cuda_path.hpp src/glrt.hpp src/cudart.hpp phosphor_params.h phosphor_half.h kernels/phosphor.cu
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDCUDA) $(LDGL)

clean:
	rm -rf bin
.PHONY: all toolchain clean
