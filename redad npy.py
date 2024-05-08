from pyecharts.charts import Line
from pyecharts import options as opts
import numpy as np
import webbrowser
import os

# 加载.npy文件
data = np.load('resultresult_set_100_5.npy', allow_pickle=True)

# 遍历数据集，提取并绘制真实值和预测值数据
for key in data.item():
    if 'true_set' in key:
        true_data = data.item().get(key)[0]
        predict_data = data.item().get(key.replace('true', 'predict'))[0]

        # 绘制折线图
        line = (
            Line()
            .add_xaxis(range(len(true_data)))
            .add_yaxis("True Data", true_data.flatten().tolist(),
                       is_smooth=True,
                       is_symbol_show=False,
                       label_opts=opts.LabelOpts(is_show=False)
                       )
            .add_yaxis("Predict Data", predict_data.flatten().tolist(),
                       is_smooth=True,
                       is_symbol_show=False,
                       label_opts=opts.LabelOpts(is_show=False)
                       )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(is_scale=True),
                yaxis_opts=opts.AxisOpts(is_scale=True),
                datazoom_opts=[opts.DataZoomOpts(pos_bottom="10%")],
                tooltip_opts=None,
                axispointer_opts=opts.AxisPointerOpts(is_show=False),
            )
        )

        # 保存图表为HTML文件，文件名包含数据集名称
        file_name = f"{key}_chart.html"
        line.render(file_name)

        # 自动在浏览器中打开HTML文件
        webbrowser.open('file://' + os.path.realpath(file_name))