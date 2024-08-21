# Audio_scoring
Score the audio on four aspects and generate the corresponding csv table file
###
**环境配置**
```
conda env create -f setup.yaml
conda activate score
```
###
**数据格式要求**  

目标文件夹下的音频需为wav格式，文本需为txt格式，且除参考音频外的推理音频需与文本一一对应（前缀相同）  
如下图所示：
！[image](https://github.com/valyentine/img/blob/main/image.png)
###
**评分**  

要提前建好一个用于存放评分输出结果的文件夹，默认是score_result，也可在--output_dir参数中修改  

对文件夹内所有wav文件的所有方法进行评分：
```
python score.py --aspect ALL --data_dir your/file/dir/path --refer your/reference/audio/file/path
```
对每个方面单独评分，xxx处可填UTMOS,NISQA,CER,而SECS需要--refer参数：
```
python score.py --aspect xxx --data_dir your/file/dir/path
```
```
python score.py --aspect SECS --data_dir your/file/dir/path --refer your/reference/audio/file/path
```
--mode参数可选择给单个文件打分或整个文件夹打分，建议选择默认的predict_dir，因为单个文件的CER打分我没写，另外三个是可以的  
 
--model_size参数为选择CER评分的模型大小，有tiny, base,small,medium,large五种可选，默认为base  

--deg参数为选择给单个文件打时输入的文件路径  

NISQA中的参数大多为原程序默认值
