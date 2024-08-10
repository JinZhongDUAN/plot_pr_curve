# plot_pr_curve
plot_pr_curve for mmdetection

本py文件对mmdetection原始文件进行了增强，可用于可视化单个或多个模型的PR曲线，仅需相应的参数即可。

1、parser.add_argument('--configs', nargs='+', help='list of error_analysis config file paths')
2、parser.add_argument('--results', nargs='+', help='list of prediction paths where error_analysis pkl result')
3、parser.add_argument('--labels', nargs='+', default=['SSD512', 'Faster RCNN', 'FCOS', 'Sparse RCNN', 'RetinaNet', 'Cascade RCNN'],
                        help='list of model labels')
注意，configs，results和labels集合的参数和个数必须一一对应。
