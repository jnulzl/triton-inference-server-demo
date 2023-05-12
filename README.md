
# Triton Inference Server入门

`本仓库以ResNet50分类网络为例快速入门Triton Inference Server`

## 相关环境

- `Ubuntu 20.04`

- `Docker 23.0.1`

- `2 x GTX 3090`

- `Driver Version: 530.30.02`

- `CUDA Version: 12.1`

- `triton container：22.05`:[Triton Inference Server Release 22.05](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel_22-05.html#rel_22-05)，最低驱动要求`515` 

- `TensorRT-8.2.5.1`

## Triton ResNet50服务端搭建

- 拉取镜像

```shell
docker pull nvcr.io/nvidia/tritonserver:22.05-py3
```

- 下载`TensorRT-8.2.5.1`并在`$PWD`目录下组织如下结构

```shell
triton_demo
|-- triton_client
`-- triton_server
    `-- TensorRT-8.2.5.1.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz
```

- 创建并进入容器进行操作配置

```shell
╰─➤ docker run --gpus=all -it  -v "$PWD/triton_demo":"/root/" -p 8100:8100 -p 8101:8101 -p 8102:8102 --name triton_server_demo --ipc=host [YOUR_IMAGE_ID]

╰─➤ cd triton_server

# For using trtexec
╰─➤ export PATH=$PWD/TensorRT-8.2.5.1/bin:$PATH

# For inspect tensorrt model info
╰─➤ python3 -m pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com

╰─➤ pip3 install $PWD/TensorRT-8.2.5.1/python/tensorrt-8.2.5.1-cp38-none-linux_x86_64.whl
```

- 获取`ONNX`模型

  1、从`PyTorch`直接导出

```python
import torch
import torchvision.models as models

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

# load model; We are going to use a pretrained resnet model
model = models.resnet50(pretrained=True).eval()
x = torch.randn(1, 3, 224, 224, requires_grad=True)

# Export the model
torch.onnx.export(model,                        # model being run
                  x,                            # model input (or a tuple for multiple inputs)
                  "resnet50.onnx",              # where to save the model (can be a file or file-like object)
                  export_params=True,           # store the trained parameter weights inside the model file
                  input_names = ['input'],      # the model's input names
                  output_names = ['output'],    # the model's output names
                  )
```

  2、直接从下载的`TensorRT`中获取

```shell
╰─➤ cp TensorRT-8.2.5.1/data/resnet50/ResNet50.onnx .
```

- `ONNX`转`TensorRT`模型

```shell
╰─➤ trtexec --onnx=ResNet50.onnx --saveEngine=model.plan --explicitBatch --useCudaGraph
```
- 查看`TensorRT`模型的输入和输出名

```shell
╰─➤  polygraphy inspect model model.plan --model-type=engine
[I] Loading bytes from /root/triton_server/model.plan
[I] ==== TensorRT Engine ====
    Name: Unnamed Network 0 | Explicit Batch Engine
    
    ---- 1 Engine Input(s) ----
    {gpu_0/data_0 [dtype=float32, shape=(1, 3, 224, 224)]}
    
    ---- 1 Engine Output(s) ----
    {gpu_0/softmax_1 [dtype=float32, shape=(1, 1000)]}
    
    ---- Memory ----
    Device Memory: 17254912 bytes
    
    ---- 1 Profile(s) (2 Tensor(s) Each) ----
    - Profile: 0
        Binding Index: 0 (Input)  [Name: gpu_0/data_0]    | Shapes: min=(1, 3, 224, 224), opt=(1, 3, 224, 224), max=(1, 3, 224, 224)
        Binding Index: 1 (Output) [Name: gpu_0/softmax_1] | Shape: (1, 1000)
    
    ---- 63 Layer(s) ----

```


- Triton推理服务

```shell
model_repository
`-- resnet50
    |-- 1
    |   `-- model.plan
    `-- config.pbtxt

╰─➤  tritonserver --model-repository=./model_repository --http-port 8100 --grpc-port 8101 --metrics-port 8102
```


## Triton ResNet50客户端演示

主机端通过`http`访问Triton推理服务

```shell
#安装依赖
pip install opencv-python
pip install attrdict
pip install nvidia-pyindex
pip install geventhttpclient
pip install tritonclient==2.22

#下载测试图像
╰─➤  wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"

╰─➤  python triton_client.py #其中的localhost可改为服务器IP
[b'0.548891:14' b'0.321422:92' b'0.020871:12' b'0.017956:17'
 b'0.012230:16']
```

## 参考资料

- [TensorRT to Triton](https://github.com/NVIDIA/TensorRT/tree/main/quickstart/deploy_to_triton)

- [Triton Tutorials](https://github.com/triton-inference-server/tutorials)

- [Getting Started with NVIDIA Triton](https://developer.nvidia.com/triton-inference-server/get-started)

- [NVIDIA Triton Inference Server Doc](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html)

- [Triton 从入门到精通(b站)](https://www.bilibili.com/video/BV1KS4y1v7zd/?spm_id_from=333.999.0.0)
