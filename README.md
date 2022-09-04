# mengzi_NLG
This repository uses the mengzi-t5-base model for auto marketing text generation

### 文件结构
```
.
├── data
│   └── data.json #数据集的json形式，仓库中为空
├── mengzi-t5-base #  mengzi-t5-base，可从网上直接下载
│   ├── README.md
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── spiece.model
│   └── spiece.vocab
└── mengzi_nlg
    ├── main.py
    ├── preprocess.py
    ├── save4.pt #模型文件
    └── test.py

```
### 训练参数说明
```
--data_json_path               STR           path of dataset.               Default is "../data/data.json".
--test_num                     INT           Size of test set               Default is 100。
--batch_size                   INT           Batch size.                    Default is 4.    
--key_threshold                FLOAT         Threshold for picking keywords Default is 0.5.
--device                       STR           Device for training.           Default is 'cuda'.
--gradient_accumulation_step   INT           Step for gradient accumulation Default is 8
```
### 训练命令
```
cd mengzi_nlg/
python main.py
```
### 测试命令
```
cd mengzi_nlg/
python test.py
```
### 初步效果：

inputs: 标题：2014成都车展：长安悦翔V7正式亮相|关键词：1.6L发动机, 成都车展, 长安悦翔V7, 新车尾部造型, 液晶显示屏, 倒梯形设计, 进气格栅, 上下进气口, 大灯, 铬装饰进行点缀|


outputs: 生成: 在2014成都车展上,长安悦翔V7正式亮相,新车搭载1.6L发动机。 外观方面,长安悦翔V7的进气格栅采用倒梯形设计,并加入镀铬装饰进行点缀,前大灯采用熏黑处理,并加入日间行车灯。尾部方面,新车尾部造型饱满,尾灯造型与前大灯相呼应。内饰方面,长安悦翔V7采用上深下浅的配色,并配有液晶显示屏。动力方面,长安悦翔V7搭载1.6L发动机,其最大功率为125马力,峰值扭矩为151N・m。
