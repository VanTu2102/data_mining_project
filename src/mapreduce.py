from mrjob.job import MRJob
from mrjob.step import MRStep
import datetime
import json
from mr3px.csvprotocol import CsvProtocol
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
import pandas as pd
import pickle

class MapReduce(MRJob):

    OUTPUT_PROTOCOL = CsvProtocol

    def mapper_etl(self, _, line):
        data = line.replace(", ", "-").split(",")
        if data[0] != "Mã học viên":
            data[1] = int(
                datetime.datetime.strptime(data[1], "%m/%d/%Y %H:%M").timestamp()
            )
            data[2] = int(
                datetime.datetime.strptime(data[2], "%m/%d/%Y %H:%M").timestamp()
            )
            data[5] = int(data[5]) / 1000000
            data[6] = data[6].replace(" buổi", "")
            data[6] = data[6].replace("tháng", "30")
            data[7] = data[7].replace("Không đạt", "0")
            data[7] = data[7].replace("Đạt", "1")
            if data[7] == "":
                data[7] = 0
            data[8] = data[8].replace("%", "")
            if data[8] == "":
                data[8] = 0
            data[6] = int(data[6])
            data[7] = int(data[7])
            data[8] = int(data[8])
            yield (data[0], ",".join(str(x) for x in data))
        else:
            yield (["HEADER"], ",".join(str(x) for x in data))

    def combiner_etl(self, key, lines):
        datum = [line.split(",") for line in lines]
        if key[0] != "HEADER":
            for i in range(len(datum) - 1):
                for j in range(i + 1, len(datum)):
                    if int(datum[j][2]) < int(datum[i][2]):
                        datum[i], datum[j] = datum[j], datum[i]
            for i in range(len(datum)):
                datum[i].append(i + 1)
                if i == len(datum) - 1:
                    datum[i].append(0)
                else:
                    datum[i].append(1)
                k = int(datum[i][2])
                datum[i][1] = datetime.datetime.fromtimestamp(
                    int(datum[i][1])
                ).strftime("%m/%d/%Y %H:%M")
                datum[i][2] = datetime.datetime.fromtimestamp(
                    int(datum[i][2])
                ).strftime("%m/%d/%Y %H:%M")
                datum[i][5] = float(datum[i][5])
                datum[i][6] = int(datum[i][6])
                datum[i][7] = int(datum[i][7])
                datum[i][8] = int(datum[i][8])
                if datum[i][10] != "":
                    datum[i][10] = int(datum[i][10])
                    yield (
                        [key, k],
                        [
                            datum[i][5],
                            datum[i][7],
                            datum[i][8],
                            datum[i][10],
                            datum[i][16],
                            datum[i][17],
                        ],
                    )
        # else:
        #     datum[0].append("Lần đăng ký")
        #     datum[0].append("Có tái tục không")
        #     yield (["HEADER"], [datum[0][0], datum[0][5], datum[0][7], datum[0][8], datum[0][10], datum[0][16], datum[0][17]])

    def reducer_etl(self, key, line):
        for row in line:
            yield (None, row)

    def mapper_training(self, key, line):
        yield ("KNN", line)
        yield ("SVM", line)
        yield ("Decision Tree", line)
        yield ("Random Forest", line)
        yield ("Logistic Regression", line)
        yield ("Gradient Boosting", line)
        yield ("Naive Bayes", line)

    def combine_training(self, key, lines):
        df = pd.DataFrame([[float(i) for i in t] for t in lines])
        X = pd.DataFrame(
            df,
            columns=[0, 1, 2, 3, 4],
        )
        y = pd.DataFrame(df, columns=[5])
        rskf = KFold(n_splits=10, shuffle=True)
        if key == "KNN":
            model = KNeighborsClassifier()
            X_train, X_test, y_train, y_test = train_test_split(X, y.to_numpy().flatten(), test_size=0.5)
            # Huấn luyện mô hình
            model.fit(X_train, y_train)
            # đánh giá trên tệp validation
            score = cross_val_score(model, X_train, y_train, cv=5)
            yield (
                [key, str(pickle.dumps(model))], score.mean()
            )

        if key == "SVM":
            model = SVC()
            X_train, X_test, y_train, y_test = train_test_split(X, y.to_numpy().flatten(), test_size=0.5)
            # Huấn luyện mô hình
            model.fit(X_train, y_train)
            # đánh giá trên tệp validation
            score = cross_val_score(model, X_train, y_train, cv=5)
            yield (
                [key, str(pickle.dumps(model))], score.mean()
            )

        if key == "Decision Tree":
            model = DecisionTreeClassifier()
            X_train, X_test, y_train, y_test = train_test_split(X, y.to_numpy().flatten(), test_size=0.5)
            # Huấn luyện mô hình
            model.fit(X_train, y_train)
            # đánh giá trên tệp validation
            score = cross_val_score(model, X_train, y_train, cv=5)
            yield (
                [key, str(pickle.dumps(model))], score.mean()
            )

        if key == "Random Forest":
            model = RandomForestClassifier()
            X_train, X_test, y_train, y_test = train_test_split(X, y.to_numpy().flatten(), test_size=0.5)
            # Huấn luyện mô hình
            model.fit(X_train, y_train)
            # đánh giá trên tệp validation
            score = cross_val_score(model, X_train, y_train, cv=5)
            yield (
                [key, str(pickle.dumps(model))], score.mean()
            )

        if key == "Logistic Regression":
            model = LogisticRegression()
            X_train, X_test, y_train, y_test = train_test_split(X, y.to_numpy().flatten(), test_size=0.45)
            # Huấn luyện mô hình
            model.fit(X_train, y_train)
            # đánh giá trên tệp validation
            score = cross_val_score(model, X_train, y_train, cv=5)
            yield (
                [key, str(pickle.dumps(model))], score.mean()
            )

        if key == "Gradient Boosting":
            model = GradientBoostingClassifier()
            X_train, X_test, y_train, y_test = train_test_split(X, y.to_numpy().flatten(), test_size=0.25)
            # Huấn luyện mô hình
            model.fit(X_train, y_train)
            # đánh giá trên tệp validation
            score = cross_val_score(model, X_train, y_train, cv=5)
            yield (
                [key, str(pickle.dumps(model))], score.mean()
            )

        if key == "Naive Bayes":
            model = GaussianNB()
            X_train, X_test, y_train, y_test = train_test_split(X, y.to_numpy().flatten(), test_size=0.95)
            # Huấn luyện mô hình
            model.fit(X_train, y_train)
            # đánh giá trên tệp validation
            for train_index, test_index in rskf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = (
                    y.iloc[train_index].values.ravel(),
                    y.iloc[test_index].values.ravel(),
                )
                model.partial_fit(X_train, y_train)
            # đánh giá trên tệp validation
            score = cross_val_score(model, X_train, y_train, cv=5)
            yield (
                [key, str(pickle.dumps(model))], score.mean()
            )

    def reducer_training(self, key, line):
        
        yield (key[0], min(line))

    def steps(self):
        return [
            MRStep(
                mapper=self.mapper_etl,
                combiner=self.combiner_etl,
                reducer=self.reducer_etl,
            ),
            MRStep(
                mapper=self.mapper_training,
                combiner=self.combine_training,
                reducer=self.reducer_training,
            ),
        ]


if __name__ == "__main__":
    MapReduce.run()
