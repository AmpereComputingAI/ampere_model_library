from sklearn.metrics import roc_auc_score

from utils.recommendation.DeepCTR.deepctr.models import ESMM
from utils.recommendation.census_income import CensusIncome


def train(batch_size=256):
    dataset = CensusIncome(batch_size, "adult.csv")
    model = ESMM(dataset.dnn_feature_columns, tower_dnn_hidden_units=[], task_types=['binary', 'binary'],
                 task_names=['label_income', 'label_marital'])
    model.compile("adam", loss=["binary_crossentropy", "binary_crossentropy"],
                  metrics=['binary_crossentropy'], )

    model.fit(dataset.train_model_input,
              [dataset.train_set['label_income'].values, dataset.train_set['label_marital'].values],
              batch_size=batch_size, epochs=10, verbose=2, validation_split=0.2)
    pred_ans = model.predict(dataset.test_model_input, batch_size=batch_size)

    print("test income AUC", round(roc_auc_score(dataset.test_set['label_income'], pred_ans[0]), 4))
    print("test marital AUC", round(roc_auc_score(dataset.test_set['label_marital'], pred_ans[1]), 4))

    filename = "esmm"
    model.save(filename)
    print(f"Saved as: {filename} (SavedModel format)")


if __name__ == "__main__":
    train()
