import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from apyori import apriori, load_transactions


def add_col(
    df, column
):  # Thêm cột lần đăng ký là lần bao nhiêu và nhãn dữ liệu có tái tục hay không tương ứng với mỗi dòng dữ liệu
    for i in range(len(df) - 1):
        for j in range(len(df) - i - 1):
            if datetime.datetime.strptime(
                df.iloc[j][column], "%Y-%m-%d"
            ) > datetime.datetime.strptime(df.iloc[j + 1][column], "%Y-%m-%d"):
                df.iloc[j], df.iloc[j + 1] = df.iloc[j + 1], df.iloc[j]
    column_index = [(i + 1) for i in range(len(df))]
    df.insert(len(df.columns), "Lần đăng ký", column_index)
    column_label = []
    for i in range(len(df) - 1):
        column_label.append(1)
    column_label.append(0)
    df.insert(len(df.columns), "Có tái tục không", column_label)
    return df


def apri(df, threshold=0.2):
    new_data = []
    last_id = None
    for j in range(len(df)):
        if df.iloc[j]["Mã học viên"] != last_id:
            last_id = df.iloc[j]["Mã học viên"]
            new_data.append(
                {
                    "Mã học viên": last_id,
                    "Khóa học": [df.iloc[j]["Khóa học"]],
                    "Người đăng ký": [df.iloc[j]["Người đăng ký"]],
                    "Khuyến mãi (khoá tiếp theo))": [
                        df.iloc[j]["Khuyến mãi (khoá tiếp theo))"]
                    ],
                    "Số kỹ năng học viên muốn đăng ký(nghe, nói, đọc, viết)": [
                        df.iloc[j][
                            "Số kỹ năng học viên muốn đăng ký(nghe, nói, đọc, viết)"
                        ]
                    ],
                    "Mã giảng viên": [df.iloc[j]["Mã giảng viên"]],
                    "Số kỹ năng khoá học": [df.iloc[j]["Số kỹ năng khoá học"]],
                }
            )
        else:
            new_data[-1]["Khóa học"].append(df.iloc[j]["Khóa học"])
            new_data[-1]["Người đăng ký"].append(df.iloc[j]["Người đăng ký"])
            new_data[-1]["Khuyến mãi (khoá tiếp theo))"].append(
                df.iloc[j]["Khuyến mãi (khoá tiếp theo))"]
            )
            new_data[-1][
                "Số kỹ năng học viên muốn đăng ký(nghe, nói, đọc, viết)"
            ].append(
                df.iloc[j]["Số kỹ năng học viên muốn đăng ký(nghe, nói, đọc, viết)"]
            )
            new_data[-1]["Mã giảng viên"].append(df.iloc[j]["Mã giảng viên"])
            new_data[-1]["Số kỹ năng khoá học"].append(
                df.iloc[j]["Số kỹ năng khoá học"]
            )
    new_data = {
        "Khóa học": [d["Khóa học"] for d in new_data],
        "Người đăng ký": [d["Người đăng ký"] for d in new_data],
        "Khuyến mãi (khoá tiếp theo))": [
            d["Khuyến mãi (khoá tiếp theo))"] for d in new_data
        ],
        "Số kỹ năng học viên muốn đăng ký(nghe, nói, đọc, viết)": [
            d["Số kỹ năng học viên muốn đăng ký(nghe, nói, đọc, viết)"]
            for d in new_data
        ],
        "Mã giảng viên": [d["Mã giảng viên"] for d in new_data],
        "Số kỹ năng khoá học": [d["Số kỹ năng khoá học"] for d in new_data],
    }
    d = {}
    for key, value in new_data.items():
        arr = []
        for item in list(apriori(value, min_support=0.0045, min_confidence=0.05, min_lift=1.1)):
            print(item[0])
            # first index of the inner list
            # Contains base item and add item
            pair = item[0] 
            items = [x for x in pair]
            print("Rule: " + items[0] + " -> " + items[1])

            #second index of the inner list
            print("Support: " + str(item[1]))

            #third index of the list located at 0th
            #of the third index of the inner list

            print("Confidence: " + str(item[2][0][2]))
            print("Lift: " + str(item[2][0][3]))
            print("=====================================")
            arr.append([str(item[0]), str(items[0] + " -> " + items[1]), str(item[1]), str(item[2][0][2]), str(item[2][0][3])])
        if(len(arr)>0):
            d[key] = arr
            print(pd.DataFrame(arr))
    writer = pd.ExcelWriter('./results/association_rule.xlsx')
    for key, value in d.items():
        # Write each DataFrame to a separate sheet
        pd.DataFrame(value, columns=['set', 'Rule', 'Support', 'Confidence', 'Lift']).to_excel(writer, sheet_name=key, index=False)
    writer.save()

def etl(path):
    # Mở file data
    data_df = pd.read_csv(path, delimiter=",")

    # Transform data chuyển dữ liệu về số
    km = {"15%": 15, "8%": 8, "5%": 5, np.nan: np.nan}
    data_df["Khuyến mãi (khoá tiếp theo))"] = data_df[
        "Khuyến mãi (khoá tiếp theo))"
    ].map(km)
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
    
    # apri(last_data)

    # print(last_data)
    # print(data_return.describe())
    # data_return.describe().to_excel("./results/data_return_describe.xlsx", index=True)
    # print(data_return.describe(include=['object']))
    # data_return.describe(include=['object']).to_excel("./results/data_return_describe_obj.xlsx", index=True)

    # print(data_non_return.describe())
    # data_non_return.describe().to_excel("./results/data_non_return_describe.xlsx", index=True)
    # print(data_non_return.describe(include=['object']))
    # data_non_return.describe(include=['object']).to_excel("./results/data_non_return_describe_obj.xlsx", index=True)

    # plt.figure(figsize=(16,6))
    # plt.hist([data_return["Khóa học"], data_non_return["Khóa học"]], bins=10,  density=True
    #          , histtype='bar', stacked=False, color=['cyan', 'green'], edgecolor='black')
    # plt.legend(['Có tái tục', 'Không tái tục'])
    # plt.savefig("results/images/hist_data.png")
    # plt.clf()

    # plt.figure(figsize=(6,6))
    # plt.hist([data_return["Lần đăng ký"], data_non_return["Lần đăng ký"]], bins=10
    #          , histtype='bar', stacked=False, color=['cyan', 'green'], edgecolor='black')
    # plt.legend(['Có tái tục', 'Không tái tục'])
    # plt.savefig("results/images/hist_data_2.png")
    # plt.clf()

    # plt.figure(figsize=(6,6))
    # plt.hist([data_return["Học phí"], data_non_return["Học phí"]], bins=10
    #          , histtype='bar', stacked=False, color=['cyan', 'green'], edgecolor='black')
    # plt.legend(['Có tái tục', 'Không tái tục'])
    # plt.savefig("results/images/hist_data_3.png")
    # plt.clf()

    # last_data.boxplot(column=[
    #         "Lần đăng ký",
    #         "Đánh giá giảng viên",
    #         "Học phí",
    #         "Kết quả sau khi học",
    #         "Khuyến mãi (khoá tiếp theo))"], by='Có tái tục không', fontsize=10, grid=True, figsize=(10,6))
    # plt.savefig("results/images/boxplot.png")
    # plt.clf()

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
