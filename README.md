# API with PCA Clustering

## API Usage

### Start the API Server

Navigate to the backend directory and start the API server with the following commands:

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Endpoints Documentation

| Endpoint               | Method | Description                                                           | Success Response                                                               | Error Response                                                          |
| ---------------------- | ------ | --------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| `/plot_pca/`           | GET    | PCA Clustering on Mall Customers Data                                 | Returns JSON object containing original data and a base64 encoded plot of PCA. | 500 Internal Server Error for file not found or model errors.           |
| `/plot_income_kmeans/` | GET    | K-Means Clustering on Annual Income and Spending Score                | JSON with clustered data, cluster count, MSE, and a base64 encoded plot.       | 500 Internal Server Error for data access or calculation errors.        |
| `/plot_age_kmeans/`    | GET    | K-Means Clustering on Age and Spending Score                          | JSON with clustered data, cluster count, MSE, and a base64 encoded plot.       | 500 Internal Server Error for data handling or processing issues.       |
| `/plot_income_gmm/`    | GET    | Gaussian Mixture Model Clustering on Annual Income and Spending Score | JSON with clustered data, cluster count, BIC, AIC, and a base64 encoded plot.  | 500 Internal Server Error for model failures or incorrect data inputs.  |
| `/plot_age_gmm/`       | GET    | Gaussian Mixture Model Clustering on Age and Spending Score           | JSON with clustered data, cluster count, BIC, AIC, and a base64 encoded plot.  | 500 Internal Server Error if there are problems with the model or data. |

## FrontEnd Usage

### Start the Frontend Server

Navigate to the frontend directory and start the server with:

```bash
cd frontend
python -m http.server 8080
```

## Training Usage

### Run Training Scripts

Navigate to the training directory and execute the training scripts:

```bash
cd training
python kmeans_and_GMM.py
python pca.py
```
