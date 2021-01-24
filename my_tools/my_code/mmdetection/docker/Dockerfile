ARG PYTORCH="1.2"
ARG CUDA="10.0"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6

# Install mmdetection
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN git clone https://linux.xidian.edu.cn/git/dfchen/mmdetection.git /mmdetection
WORKDIR /mmdetection
RUN git clone https://linux.xidian.edu.cn/git/dfchen/mmcv.git /mmcv
RUN pip install -r /mmcv/requirements.txt
RUN pip install -e /mmcv
RUN pip install -r requirements/build.txt
RUN pip install -r requirements/optional.txt
RUN pip install -r requirements/runtime.txt
RUN pip install -r requirements/tests.txt
RUN pip install "git+https://linux.xidian.edu.cn/git/dfchen/cocoapi.git#subdirectory=PythonAPI"
ENV FORCE_CUDA="1"
RUN pip install -v -e .

