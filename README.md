# PoER
Potential Energy Ranking for Out of Distribution Detection


data/ -- save the trainning data
log/ -- training log
data.py -- data preprocessing code
models.py -- model architecture file
train.py -- training script
eval.py -- eval script, all kinds of evaluation methods
losses.py -- all kinds of loss function 
ood.py -- all kinds of ood confidence computer


## 拓展指引

- 拓展ood confidence 的计算方式，请修改ood.py 中的 ReconPostProcessor模块，或者增加一个新的类，目前就按方式是直接按照Recon loss来作为ood confidence，如果添加新的类，需要在train.py 的 eval_ood()，将新的PostProcessor作为参数传进去。

```
 result = evaluator.eval_ood(val_loader, ood_data_loaders=ood_loader, post_processor=conf_processor, method='full')
```


- 拓展训练损失，修改 losses.py 中的损失函数即可。

- 拓展模型，请直接修改models.py，如果models的返回值的个数或者顺序发生变动，请修改PostProcessor。