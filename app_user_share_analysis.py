# -*- coding: utf-8 -*-
# APP 用户分享行为分析 - 完整代码

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ========== 1. 读数据 ==========
df = pd.read_csv("某APP用户信息数据.csv")
print("数据规模:", df.shape)

# ========== 2. 数据清洗 ==========
df = df.dropna()   # ① 删缺失值
df = df[(df["不愿分享概率"] >= 0) & (df["不愿分享概率"] <= 1) &
        (df["愿意分享概率"] >= 0) & (df["愿意分享概率"] <= 1)]   # ② 删异常值（概率超 0~1）
q99 = df["在线时长/分钟"].quantile(0.99)
df = df[df["在线时长/分钟"] <= q99]   # ③ 删极端值（在线时长 > 99% 分位）
print("清洗后规模:", df.shape)

# ========== 3. 对比分析 ==========
print("\n分享 vs 未分享人数：")
print(df["是否点击分享"].value_counts())
print("\n两类用户在线时长对比：")
print(df.groupby("是否点击分享")["在线时长/分钟"].agg(["mean", "median"]))
print("\n两类用户分享概率对比：")
print(df.groupby("是否点击分享")[["愿意分享概率", "不愿分享概率"]].mean())

# ========== 4. 逻辑回归建模 ==========
df["是否点击分享"] = df["是否点击分享"].map({"T": 1, "F": 0})
X = df[["愿意分享概率", "不愿分享概率", "在线时长/分钟"]]
y = df["是否点击分享"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n逻辑回归准确率:", round(accuracy_score(y_test, y_pred), 4))
