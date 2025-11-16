## 构建数据库
```shell
python3 main.py --input ./media_dataset.json --db ./graph.sqlite
```
## 输出可视化图
```shell
python3 viz.py --db ./graph.sqlite --mode topk --k 100 --with-labels --layout spring --seed 42 --out viz_top100.png --dpi 200
```
