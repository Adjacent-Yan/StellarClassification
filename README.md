# StellarClassification

## EDA Processing (EDA.py)
Make sure the `star_classification.csv` file is in the same folder as the `EDA.py` file.
Run `EDA.py` which analyzes the star_classification.csv data downloaded from the SDSS DR17 Dataset on Kaggle by 
fedesoriano on Kaggle.
## KNN Model (knn.ipynb)
Make sure the `star_classification.csv file` is in the same folder `knn.ipynb`.
Run `knn.ipynb` which runs the KNN model training process, validation,  final test, and  displays its confusion matrix.
## Random Forest Model (random_forest.ipynb)
Make sure the `star_classification.csv` file is in the same folder as `random_rainforest.ipynb`.
Run `random_forest.ipynb` which runs the random forest training process, validation, testing, ablation study, and the confusion matrices.
## MLP Model (mlp.ipynb)
Make sure the `star_classification.csv` file is in the same folder as `mlp.ipynb`.
Run `mlp.ipynb` which runs the mlp classifier training process, validation, hyperparameter tuning, testing, and displays the final confusion matrix.
## LSTM Model (lstm.ipynb)

For our recurrent neural network model, we used a Long Short-Term Memory (LSTM) model to classify objects as GALAXY, QSO, or STAR. The model uses the u, g, r, i, and z brightness measurements as a sequence based on wavelength. Redshift is kept as a separate input and is later combined with the LSTM output.

The LSTM model can be found in `lstm.ipynb`.

### How to Run the LSTM

First, install the required packages:

```bash
pip install -r requirements.txt
```

Make sure `star_classification.csv` is in the main project folder.

Then open `lstm.ipynb` in VS Code and run the cells from top to bottom.

The notebook will load and clean the data, split it into training, validation, and testing sets, train the LSTM model, and show the final results and graphs.

### LSTM Results

- Validation Accuracy: 96.43%
- Validation Macro F1: 95.97%
- Test Accuracy: 96.37%
- Test Macro F1: 95.91%

One limitation of the LSTM is that our data is not actually a time sequence. The u, g, r, i, and z measurements are ordered by wavelength instead of time. We used the LSTM to see if treating the brightness measurements as a sequence could help the model learn patterns between the different bands.