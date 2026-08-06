# StellarClassification


## LSTM Model

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