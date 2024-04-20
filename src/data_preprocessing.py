import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis

def add_col(df, column): # Thêm cột lần đăng ký là lần bao nhiêu và nhãn dữ liệu có tái tục hay không tương ứng với mỗi dòng dữ liệu
    for i in range(len(df) - 1):
        for j in range(len(df) - i - 1):
            if datetime.datetime.strptime(
                df.iloc[j][column], "%m/%d/%Y %H:%M"
            ) > datetime.datetime.strptime(df.iloc[j + 1][column], "%m/%d/%Y %H:%M"):
                df.iloc[j], df.iloc[j + 1] = df.iloc[j + 1], df.iloc[j]
    column_index = [(i + 1) for i in range(len(df))]
    df.insert(len(df.columns), "Lần đăng ký", column_index)
    column_label = []
    for i in range(len(df) - 1):
        column_label.append(1)
    column_label.append(0)
    df.insert(len(df.columns), "Có tái tục không", column_label)
    return df


def etl():
    # Mở file data
    data_df = pd.read_csv("./data/data.csv", delimiter=",")

    # Transform data chuyển dữ liệu về số
    km = {"15%": 15, "8%": 8, "5%": 5, np.nan: np.nan}
    data_df["Khuyến mãi (khoá tiếp theo))"] = data_df["Khuyến mãi (khoá tiếp theo))"].map(
        km
    )
    kq = {"Đạt": 1, "Không đạt": 0, np.nan: np.nan}
    data_df["Kết quả sau khi học"] = data_df["Kết quả sau khi học"].map(kq)
    
    data_df["Học phí"] = data_df["Học phí"] / 1000000

    # Loại bỏ các cột thừa
    data_df.drop(
        [
            # "Địa chỉ(quê quán gv)",
            "Số điện thoại gv",
            # "Khóa học",
            # "Giới tính",
            # "Ngày sinh gv",
            # "Họ và tên giảng viên",
            # "Thời lượng",
        ],
        axis=1,
        inplace=True,
    )

    
    # Nhóm dữ liệu theo mã học viên và giữ lại các cột cần thiết
    data_df = data_df.groupby("Mã học viên")
    last_data = pd.concat([add_col(group, "Ngày đăng ký") for name, group in data_df])
    
    data_return = last_data.loc[last_data['Có tái tục không'] == 1]
    data_non_return = last_data.loc[last_data['Có tái tục không'] == 0]
    
    print(last_data)
    print(data_return.describe())
    data_return.describe().to_excel("./results/data_return_describe.xlsx", index=True)
    print(data_return.describe(include=['object']))
    data_return.describe(include=['object']).to_excel("./results/data_return_describe_obj.xlsx", index=True)
    
    print(data_non_return.describe())
    data_non_return.describe().to_excel("./results/data_non_return_describe.xlsx", index=True)
    print(data_non_return.describe(include=['object']))
    data_non_return.describe(include=['object']).to_excel("./results/data_non_return_describe_obj.xlsx", index=True)
    
    plt.figure(figsize=(16,6))
    plt.hist([data_return["Khóa học"], data_non_return["Khóa học"]], bins=10,  density=True
             , histtype='bar', stacked=False, color=['cyan', 'green'], edgecolor='black')
    plt.legend(['Có tái tục', 'Không tái tục'])
    plt.savefig("results/images/hist_data.png")
    plt.clf()
    
    plt.figure(figsize=(6,6))
    plt.hist([data_return["Lần đăng ký"], data_non_return["Lần đăng ký"]], bins=10
             , histtype='bar', stacked=False, color=['cyan', 'green'], edgecolor='black')
    plt.legend(['Có tái tục', 'Không tái tục'])
    plt.savefig("results/images/hist_data_2.png")
    plt.clf()
    
    plt.figure(figsize=(6,6))
    plt.hist([data_return["Học phí"], data_non_return["Học phí"]], bins=10
             , histtype='bar', stacked=False, color=['cyan', 'green'], edgecolor='black')
    plt.legend(['Có tái tục', 'Không tái tục'])
    plt.savefig("results/images/hist_data_3.png")
    plt.clf()
    
    last_data.boxplot(column=[
            "Lần đăng ký",
            "Đánh giá giảng viên",
            "Học phí",
            "Kết quả sau khi học",
            "Khuyến mãi (khoá tiếp theo))"], by='Có tái tục không', fontsize=10, grid=True, figsize=(10,6))
    plt.savefig("results/images/boxplot.png")
    plt.clf()
    
    # sns.histplot(last_data, x='Đánh giá giảng viên', hue="Có tái tục không", multiple="dodge", shrink=.5, binwidth=1)
    # plt.show()
    last_data = last_data[
        [
            "Mã học viên",
            "Có tái tục không",
            "Lần đăng ký",
            "Đánh giá giảng viên",
            "Học phí",
            "Kết quả sau khi học",
            "Khuyến mãi (khoá tiếp theo))",
        ]
    ].dropna()
    
    # sns.boxplot(data=data_return, color='teal', palette='rocket')
    # # sns.boxplot(data=data_non_return, color='tomato', palette='rocket')

    # # Thêm tiêu đề
    # plt.title('Biểu đồ boxplot')

    # # Hiển thị biểu đồ
    # plt.show()
    # # Tính hệ số tương quan Pearson
    # correlation = last_data.corr()
    
    # # Vẽ biểu đồ ma trận tương quan
    # sns.heatmap(correlation, annot=True)
    # plt.show()

    return last_data