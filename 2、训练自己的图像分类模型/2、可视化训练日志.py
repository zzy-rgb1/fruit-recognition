import pandas as pd
import matplotlib.pyplot as plt

# 设置Matplotlib中文字体
# windows操作系统
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 载入训练日志表格
df_train = pd.read_csv('训练日志-训练集.csv')
df_test = pd.read_csv('训练日志-测试集.csv')

# 训练集损失函数
plt.figure(figsize=(16, 8))
x = df_train['batch']
y = df_train['train_loss']
plt.plot(x, y, label='训练集')
plt.tick_params(labelsize=20)
plt.xlabel('batch', fontsize=20)
plt.ylabel('损失函数', fontsize=20)
plt.title('训练集损失函数', fontsize=25)
plt.savefig('./图表/训练集损失函数.pdf', dpi=120, bbox_inches='tight')
plt.show()

# 训练集准确率
plt.figure(figsize=(16, 8))
x = df_train['batch']
y = df_train['train_accuracy']
plt.plot(x, y, label='训练集')
plt.tick_params(labelsize=20)
plt.xlabel('batch', fontsize=20)
plt.ylabel('准确率', fontsize=20)
plt.title('训练集准确率', fontsize=25)
plt.savefig('图表/训练集准确率.pdf', dpi=120, bbox_inches='tight')
plt.show()

# 测试集的损失函数
plt.figure(figsize=(16, 8))
x = df_test['epoch']
y = df_test['test_loss']
plt.plot(x, y, label='测试集')
plt.tick_params(labelsize=20)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('损失函数', fontsize=20)
plt.title('测试集损失函数', fontsize=25)
plt.savefig('图表/测试集损失函数.pdf', dpi=120, bbox_inches='tight')
plt.show()

# 测试集评估指标
from matplotlib import colors as mcolors
import random
random.seed(124)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred', 'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan', 'dodgerblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet', 'mediumorchid', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink']
markers = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11]
linestyle = ['--', '-.', '-']
def get_line_arg():
    '''
    随机产生一种绘图线型
    '''
    line_arg = {}
    line_arg['color'] = random.choice(colors)
    # line_arg['marker'] = random.choice(markers)
    line_arg['linestyle'] = random.choice(linestyle)
    line_arg['linewidth'] = random.randint(1, 4)
    # line_arg['markersize'] = random.randint(3, 5)
    return line_arg

metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1-score']

plt.figure(figsize=(16, 8))
x = df_test['epoch']
for y in metrics:
    plt.plot(x, df_test[y], label=y, **get_line_arg())
plt.tick_params(labelsize=20)
plt.ylim([0, 1])
plt.xlabel('epoch', fontsize=20)
plt.ylabel('评估指标', fontsize=20)
plt.title('测试集分类评估指标', fontsize=25)
plt.savefig('图表/测试集分类评估指标.pdf', dpi=120, bbox_inches='tight')
plt.legend(fontsize=20)
plt.show()
