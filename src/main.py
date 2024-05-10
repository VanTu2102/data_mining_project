import os
import data_preprocessing
import model_training
import model_evaluation
import visualization
import pandas as pd

class Main():

    def run(self, path):
        data = data_preprocessing.etl(path)
        # models = model_training.train(data)
        
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
        return model_evaluation.report(
        {
            "data": {
                "X_test": X,
                "y_test": y,
                "full": data,
            },
        }
        )
