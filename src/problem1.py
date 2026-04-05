import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import shap
warnings.filterwarnings('ignore')

# 设置中文和英文字体，保证画图时不会乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = False

class NIPTAnalyzer:
    def __init__(self):
        self.data = None  # 原始数据
        self.processed_data = None  # 预处理后的数据
        self.model_results = {}  # 保存模型结果

    def load_data(self, file_path):
        """读取Excel数据文件"""
        print("=== 步骤1：数据读取 ===")
        try:
            # 用pandas读取excel文件
            self.data = pd.read_excel(file_path)
            print(f"成功读取数据，共{len(self.data)}条记录")
            print(f"数据列名示例：{list(self.data.columns[:10])}...")
            return True
        except Exception as e:
            print(f"数据读取失败：{e}")
            return False

    def preprocess_data(self):
        """对数据进行预处理，比如格式转换、缺失值处理、异常值去除"""
        print("\n=== 步骤2：数据预处理 ===")
        df = self.data.copy()

        # 将“孕周”从字符串转成数值，比如“12w+3”变成12+3/7
        def parse_gestational_week(week_str):
            if pd.isna(week_str): return None
            try:
                if 'w' in str(week_str):
                    parts = str(week_str).replace('w', '').split('+')
                    weeks = float(parts[0])
                    days = float(parts[1]) if len(parts) > 1 else 0
                    return weeks + days / 7
                else:
                    return float(week_str)
            except:
                return None

        df['孕周数值'] = df['检测孕周'].apply(parse_gestational_week)

        # 把年龄、身高、体重、BMI、Y浓度都转成数值
        numeric_columns = ['年龄', '身高', '体重', '孕妇BMI', 'Y染色体浓度']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 有效数据筛选：这些关键字段不能缺失
        valid_mask = (
            df['Y染色体浓度'].notna() &
            (df['Y染色体浓度'] > 0) &
            df['孕周数值'].notna() &
            df['孕妇BMI'].notna() &
            df['年龄'].notna()
        )
        df_clean = df[valid_mask].copy()

        # 用IQR方法去掉极端异常值
        def remove_outliers_iqr(data, column, factor=1.5):
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

        for col in ['Y染色体浓度', '孕妇BMI', '孕周数值']:
            df_clean = remove_outliers_iqr(df_clean, col)

        print(f"原始数据：{len(self.data)}条")
        print(f"有效数据（去掉缺失）：{len(df[valid_mask])}条")
        print(f"去掉异常值后：{len(df_clean)}条")
        print(f"数据利用率：{len(df_clean) / len(self.data) * 100:.1f}%")

        # 给BMI分个类，方便后面画图分析
        df_clean['BMI分类'] = pd.cut(df_clean['孕妇BMI'], bins=[0, 18.5, 25, 30, 35, float('inf')],
                                    labels=['偏瘦', '正常', '超重', '肥胖', '重度肥胖'])
        # 判断Y染色体浓度是否达到0.04的阈值
        df_clean['Y浓度达标'] = (df_clean['Y染色体浓度'] >= 0.04).astype(int)

        print(f"不同孕妇数量：{df_clean['孕妇代码'].nunique()}人")
        print(f"有多次检测的孕妇：{df_clean.groupby('孕妇代码').size().gt(1).sum()}人")

        self.processed_data = df_clean
        print("数据预处理完成！\n")
        return df_clean

    def exploratory_analysis(self):
        """做一些基础的统计和画图，直观看看数据的分布情况"""
        print("=== 步骤3：探索性数据分析 ===")
        df = self.processed_data

        # 看几个关键变量的统计值（平均数、标准差、最大最小值等）
        key_vars = ['Y染色体浓度', '孕周数值', '孕妇BMI', '年龄']
        desc_stats = df[key_vars].describe()
        print("关键变量描述性统计：")
        print(desc_stats.round(4))

        # 看看多少人Y染色体浓度达到阈值
        threshold_stats = df['Y浓度达标'].value_counts()
        reach_rate = threshold_stats[1] / len(df) * 100 if 1 in threshold_stats else 0
        print(f"\nY染色体浓度达标情况：")
        print(f"达标(≥4%)：{threshold_stats.get(1, 0)}例 ({reach_rate:.1f}%)")
        print(f"未达标(<4%)：{threshold_stats.get(0, 0)}例 ({100 - reach_rate:.1f}%)")

        # 画Y染色体浓度直方图
        plt.figure(figsize=(8, 5))
        plt.hist(df['Y染色体浓度'], bins=50, alpha=0.7, color='#FF6B6B', edgecolor='black')
        plt.axvline(x=0.04, color='#4ECDC4', linestyle='--', linewidth=2, label='4%阈值')
        plt.title('Y染色体浓度分布')
        plt.xlabel('Y染色体浓度')
        plt.ylabel('频数')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.show()

        return desc_stats

    def correlation_analysis(self):
        """分析变量之间的相关性，看看谁和Y染色体浓度关系大"""
        print("=== 步骤4：相关性分析 ===")
        df = self.processed_data
        numeric_vars = ['Y染色体浓度', '孕周数值', '孕妇BMI', '年龄', '身高', '体重']

        corr_data = df[numeric_vars].corr()
        print("Pearson相关系数矩阵：")
        print(corr_data.round(4))

        # 热力图可视化
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_data, annot=True, cmap='Spectral', center=0,
                    square=True, fmt='.3f', cbar_kws={"shrink": .8})
        plt.title('相关性热图')
        plt.tight_layout()
        plt.show()

        return corr_data

    def build_regression_models(self):
        """建立一个最简单的多元线性回归模型"""
        print("=== 步骤5：回归模型建立 ===")
        df = self.processed_data

        # 自变量：孕周、BMI、年龄
        X_columns = ['孕周数值', '孕妇BMI', '年龄']
        X = df[X_columns]
        # 因变量：Y染色体浓度
        y = df['Y染色体浓度']

        # 建立模型
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # 模型表现指标
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        print("线性回归模型结果：")
        print(f"截距: {model.intercept_:.6f}")
        for i, coef in enumerate(model.coef_):
            print(f"{X_columns[i]}系数: {coef:.6f}")
        print(f"R²: {r2:.4f}")
        print(f"MSE: {mse:.6f}")

        equation = f"Y染色体浓度 = {model.intercept_:.6f}"
        for i, coef in enumerate(model.coef_):
            equation += f" + ({coef:.6f}) × {X_columns[i]}"
        print(f"\n回归方程：\n{equation}")

        self.model_results = {
            'linear': {'model': model, 'r2': r2, 'mse': mse, 'y_pred': y_pred, 'X': X}
        }
        return self.model_results

    def model_validation(self):
        """验证模型效果（这里只保留简单直观的部分）"""
        print("=== 步骤6：模型验证 ===")
        df = self.processed_data
        y_actual = df['Y染色体浓度']
        y_pred = self.model_results['linear']['y_pred']
        model = self.model_results['linear']['model']
        X = self.model_results['linear']['X']

        # 画实际值 vs 预测值的散点图
        plt.figure(figsize=(8, 5))
        plt.scatter(y_actual, y_pred, alpha=0.6, s=20, color='#FF6B6B')
        plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2, label='完美预测线')
        plt.xlabel('实际Y染色体浓度')
        plt.ylabel('预测Y染色体浓度')
        plt.title(f'预测vs实际 (R²={self.model_results["linear"]["r2"]:.4f})')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # 用SHAP来解释特征重要性
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, feature_names=['孕周数值', '孕妇BMI', '年龄'], show=False)
        plt.title("SHAP特征重要性分析")
        plt.tight_layout()
        plt.show()

        return {
            'r2': self.model_results['linear']['r2'],
            'mse': self.model_results['linear']['mse']
        }

def main():
    analyzer = NIPTAnalyzer()
    file_path = "男胎检测.xlsx"  # 数据文件路径
    if analyzer.load_data(file_path):
        analyzer.preprocess_data()
        analyzer.exploratory_analysis()
        analyzer.correlation_analysis()
        analyzer.build_regression_models()
        analyzer.model_validation()
        print("\n分析完成！模型结果已保存。")
        return analyzer
    else:
        print("分析失败，请检查数据文件路径！")
        return None

if __name__ == "__main__":
    result = main()
