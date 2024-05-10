from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import openpyxl
import json
import pickle


def report(data):
        
    reports = []
    y_true = data["data"]["y_test"]
    #Mở file và sử dụng mô hình
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
        for name, m in model.items():
            y_pred = m.predict(data["data"]["X_test"])
            data["data"]["X_test"].insert(len(data["data"]["X_test"].columns), "Dự đoán", y_pred)
            data["data"]["X_test"].insert(len(data["data"]["X_test"].columns), "Có tái tục không", y_true)
            data["data"]["full"].insert(len(data["data"]["X_test"].columns), "Dự đoán", y_pred)
            kq = {1: "Có", 0:"Không"}
            data["data"]["X_test"]["Có tái tục không"] = data["data"]["X_test"]["Có tái tục không"].map(kq)
            data["data"]["X_test"]["Dự đoán"] = data["data"]["X_test"]["Dự đoán"].map(kq)
            data["data"]["full"]["Có tái tục không"] = data["data"]["full"]["Có tái tục không"].map(kq)
            data["data"]["full"]["Dự đoán"] = data["data"]["full"]["Dự đoán"].map(kq)
            print(data["data"]["full"])
            test = classification_report(y_true, y_pred, output_dict=True)
            reports.append({name: test})
            print(f"Name: {name}")
            print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
            print(f"precision : {precision_score(y_true, y_pred, pos_label=1)}")
            print(f"recall: {recall_score(y_true, y_pred, pos_label=1)}")
            print(f"F1 Score: {f1_score(y_true, y_pred, pos_label=1)}")
            # data["data"]["full"].to_excel("./results/result.xlsx", index=False)
            return data["data"]["full"]

    with open("./results/classification_report.json", "w") as file:
        json.dump({"reports": reports}, file)
