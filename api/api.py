from fastapi import FastAPI, HTTPException
import uvicorn
import pandas as pd
import pickle
from pydantic import BaseModel

# Set config --> No óptimo
host = "0.0.0.0"
port = 8080
debug = False
reload = False


# Load transformer & model
try:
    with open('outputs/transformer.pickle', 'rb') as r:
        transformer = pickle.load(r)
except FileNotFoundError:
    raise FileNotFoundError("Transformer file not found.")
except pickle.UnpicklingError:
    raise ValueError("Error loading transformer file.")
    
try:
    with open('outputs/model.pickle', 'rb') as r:
        model = pickle.load(r)
except FileNotFoundError:
    raise FileNotFoundError("Model file not found.")
except pickle.UnpicklingError:
    raise ValueError("Error loading model file.")

app = FastAPI()

class IrisPrediction(BaseModel):
    iris_type: str

@app.get("/get-iris-type", response_model=IrisPrediction)
def get_iris(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float
    ):

    # Build dataset
    data_predict = pd.DataFrame.from_dict(
        data={
            'sepal length (cm)': [sepal_length],
            'sepal width (cm)': [sepal_width],
            'petal length (cm)': [petal_length],
            'petal width (cm)': [petal_width]
        }
    )

    # Transform data
    try:
        data_transformed = transformer.transform(data_predict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transforming data: {e}")

    # Predict
    try:
        prediction = model.predict(data_transformed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")
    
    # Mapear la prediccion numerica a los nombres de las flores
    iris_species = {0: "setosa", 1: "versicolor", 2: "virginica"}
    iris_type = iris_species.get(prediction[0], "unknown")

    return IrisPrediction(iris_type=iris_type)

if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port, reload=reload)