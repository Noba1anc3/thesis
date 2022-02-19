# 基于PaddleOCR的文档方向四分类说明文档

## Introduction

本工程主要由文档方向微调、DB文字检测、文字方向分类三部分组成。采用了逆宽⾼⽐过滤、阈值宽⾼⽐过滤、和基于分数的⽂字⽅向投票的策略，在测试集上准确率达到99.53%．

## Usage

```
pip install paddlepaddle-gpu==2.0.0b0 / paddlepaddle==2.0.0b0
pip install -r requirments.txt
cd tools/infer
python predict_system.py
```

## Interface

- Interface Class : tools/infer/predict_system.py -> TextSystem
- `__init__` : `text_sys = TextSystem(utility.parse_args(), DET_MODEL_DIR, CLS_MODEL_DIR, GPU)`
  - `DET_MODEL_DIR: '../../inference/ch_ppocr_mobile_v1.1_det_infer/'` (default)
  - `CLS_MODEL_DIR: '../../inference/ch_ppocr_mobile_v1.1_cls_infer/'` (default)
  - `GPU: True` (default)
- `__call__` : `text_sys(image, cls_box_num)`
  - Input : 
    - `image` : numpy.ndarray ( w * h * 3)
    - `cls_box_num` : 用于分类的文字框数量 (default = 10)
  - Output : 文档方向
    - `0` : 上向 (文字阅读方向为自左向右)
    - `1` : 左向 (文字阅读方向为自下向上)
    - `2` : 下向 (文字阅读方向为自右向左)
    - `3` : 右向 (文字阅读方向为自上向下)

## Architecture

```
PaddleOCR
├── README.md
├── requirments.txt  // 安装依赖
│
├── configs   // 配置文件，可通过yml文件选择模型结构并修改超参
│   ├── cls   // 方向分类器相关配置文件
│   │   ├── cls_mv3.yml               // 训练配置相关，包括骨干网络、head、loss、优化器
│   │   └── cls_reader.yml            // 数据读取相关，数据读取方式、数据存储路径
│   └── det   // 检测相关配置文件
│       ├── det_db_icdar15_reader.yml // 数据读取
│       └── det_mv3_db.yml            // 训练配置
│                                  
├── ppocr            // 网络核心代码
│   ├── data         // 数据处理
│   │   ├── det   // 检测
│   │   │   ├── data_augment.py       // 数据增广操作
│   │   │   ├── dataset_traversal.py  // 数据传输，定义数据读取器，读取数据并组成batch
│   │   │   ├── db_process.py         // db 数据处理
│   │   │   ├── east_process.py       // east 数据处理
│   │   │   ├── make_border_map.py    // 生成边界图
│   │   │   ├── make_shrink_map.py    // 生成收缩图
│   │   │   ├── random_crop_data.py   // 随机切割
│   │   │   └── sast_process.py       // sast 数据处理
│   ├── postprocess   // 后处理
│   │   ├── db_postprocess.py     // DB 后处理
│   │   ├── east_postprocess.py   // East 后处理
│   │   ├── locality_aware_nms.py // nms
│   │   └── sast_postprocess.py   // sast 后处理
│   └── utils   // 工具
│       └── utility.py            // 工具函数，包含输入参数是否合法等相关检查工具
│
└── tools   // 启动工具
    └── infer                 // 基于预测引擎预测
        ├── correct.py        // 文档角度微调 
        ├── predict_cls.py      
        ├── predict_det.py
        ├── predict_system.py
        └── utility.py
```

## Appendix

### Model List

| 模型简介                | 模型名称                     | 推荐场景        | 检测模型                                                     | 方向分类模型                                                 |
| ----------------------- | ---------------------------- | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 中英文超轻量OCR模型     | ch_ppocr_mobile_v1.1_xx      | 移动端&服务器端 | [推理模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/det/ch_ppocr_mobile_v1.1_det_infer.tar) | [推理模型](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_infer.tar) |
| 中英文通用OCR模型       | ch_ppocr_server_v1.1_xx      | 服务器端        | [推理模型](https://paddleocr.bj.bcebos.com/20-09-22/server/det/ch_ppocr_server_v1.1_det_infer.tar) | [推理模型](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_infer.tar) |
| 中英文超轻量压缩OCR模型 | ch_ppocr_mobile_slim_v1.1_xx | 移动端          | [推理模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile-slim/det/ch_ppocr_mobile_v1.1_det_prune_infer.tar) | [推理模型](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_quant_infer.tar) |

### Recommended Config for GPU Environment

#### model

在GPU环境上推荐使用`mobile_server + mobile_cls`的配置

总任务平均耗时
- `mobile_server + mobile_cls` : 74.34ms
- `mobile_det + mobile_cls` : 65.54ms

总准确率
- `mobile_server + mobile_cls` : 99.67%
- `mobile_det + mobile_cls` : 99.53%

组合替换方法 : 
1. 下载[`mobile_server`](https://paddleocr.bj.bcebos.com/20-09-22/server/det/ch_ppocr_server_v1.1_det_infer.tar)
2. 将模型放在根目录下的`inference`文件夹内
3. 修改类初始化参数`DET_MODEL_DIR`为`'../../inference/ch_ppocr_server_v1.1_det_infer/'`

#### cls_box_num

以最低时间成本达到99%以上的总准确率: 推荐将`cls_box_num`配置为`5`
- 分类任务的平均耗时不足10ms
- 分类任务的最高耗时不足15ms

对耗时基本没有要求: 推荐将`cls_box_num`配置为`20`
- 分类任务的最高耗时不足50ms
- 总准确率为99.53%

对时间消耗的要求介于两者之间: 推荐使用默认配置
- 分类任务的平均耗时不足20ms
- 总任务的平均耗时不足100ms
- 总准确率为99.42%

### Recommended Config for CPU Environment

#### model

在CPU环境上推荐使用当前`mobile_det + mobile_cls`的默认配置

#### cls_box_num

- cls_det_num = 10
  - 分类任务的平均耗时降低2/3至110ms
  - 总任务的平均耗时降低1/3至558ms
  - 总准确率: 99.39%

- cls_det_num = 5
  - 分类任务平均耗时: 55ms
  - 总准确率: 98.67%



