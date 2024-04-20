from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import graphviz


def train(data):
    model_ch = {
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Naive Bayes": GaussianNB(),
    }
    max_accuracy = 0
    scores = {
        "KNN": [],
        "SVM": [],
        "Decision Tree": [],
        "Random Forest": [],
        "Logistic Regression": [],
        "Gradient Boosting": [],
        "Naive Bayes": [],
    }
    rskf = KFold(n_splits=10, shuffle=True)
    # Khởi tạo mô hình
    models = {
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Naive Bayes": GaussianNB(),
    }
    X = pd.DataFrame(
        data,
        columns=[
            "Lần đăng ký",
            "Đánh giá giảng viên",
            "Học phí",
            "Kết quả sau khi học",
            "Khuyến mãi (khoá tiếp theo))",
        ],
    )
    y = pd.DataFrame(data, columns=["Có tái tục không"])
    
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = (
            y.iloc[train_index].values.ravel(),
            y.iloc[test_index].values.ravel(),
        )
        for name, model in models.items():
            # Huấn luyện mô hình
            model.fit(X_train, y_train)

            # đánh giá trên tệp validation
            score = cross_val_score(model, X_train, y_train, cv=5)
            scores[name].append(score.mean())
            print(f"Name: {name}")
            print(f"Accuracy: {score.mean()}")
    scores = pd.DataFrame(scores)
    scores.to_excel("./results/validation.xlsx", index=True)

    # # Vẽ cây quyết định    
    # dot_data = tree.export_graphviz(models["Decision Tree"], out_file=None, feature_names=[
    #         "Lần đăng ký",
    #         "Đánh giá giảng viên",
    #         "Học phí",
    #         "Kết quả sau khi học",
    #         "Khuyến mãi (khoá tiếp theo))"], class_names=['0', '1'],filled=True, rounded=True,  
    #               special_characters=True)
    # graph = graphviz.Source(dot_data)
    # print(graph)
    # graph.view()
    # Lựa chọn mô hình có độ chính xác cao nhất trong các mô hình đã sử dụng
    max_accuracy = scores.mean().argmax()

    result = {
        "models": {scores.columns[max_accuracy]: models[scores.columns[max_accuracy]]},
        "data": {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "full": data.iloc[test_index]
        },
    }
    print(
        "Best score: ",
        cross_val_score(
            result["models"][scores.columns[max_accuracy]], X_train, y_train, cv=5
        ).mean(),
    )

    # Lưu mô hình
    with open("model.pkl", "wb") as f:
        pickle.dump(result["models"], f)
    return result
