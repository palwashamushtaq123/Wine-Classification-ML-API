from pathlib import Path 

import joblib 
import numpy as np 
from fastapi import FastAPI, HTTPException, Request 
from fastapi.responses import HTMLResponse 
from fastapi.staticfiles import StaticFiles 
from fastapi.templating import Jinja2Templates 
from pydantic import BaseModel, Field 

BASE_DIR = Path(__file__).resolve().parent 
print(f'BASE DIR:{BASE_DIR}')
model_path = BASE_DIR/'model.pkl' 

if not model_path.exists():
    raise RuntimeError("model.pkl was not found.Run 'python train.ipynb' first.")

model_bundle = joblib.load(model_path) 
model = model_bundle['model'] 
target_names = model_bundle['target_names'] 
accuracy = model_bundle['accuracy'] 
version = model_bundle['version'] 

print(type(model_bundle["accuracy"]))
print(model_bundle["accuracy"])

app = FastAPI(
    title= 'Wine Classification Web App', 
    version = model_bundle['version'], 
)

app.mount(
    '/static',
    StaticFiles(directory=BASE_DIR/'static'),
    name= 'static',
)


templates = Jinja2Templates(directory= BASE_DIR/'templates')


class WineInput(BaseModel):
    alcohol: float = Field(gt=0) 
    malic_acid: float = Field(gt=0) 
    ash: float = Field(gt=0) 
    alcalinity_of_ash: float = Field(gt=0)
    magnesium: int = Field(gt=0)
    total_phenols: float = Field(gt=0)
    flavanoids: float = Field(gt=0)
    nonflavanoid_phenols: float = Field(gt=0)
    proanthocyanins: float = Field(gt=0)
    color_intensity: float = Field(gt=0)
    hue: float = Field(gt=0)
    diluted_wines: float = Field(gt=0)
    proline: int = Field(gt=0)


@app.get('/',response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name= 'index.html',
        context={
            'model_version':model_bundle['version'],
            'model_accuracy': f"{model_bundle['accuracy'] * 100:.1f}%",
        }
    )

# @app.get('/')
# def home():
#     return{
#         "message": "Welcome to the Wine Classification API",
#         "model_version": model_bundle["version"],
#         "model_accuracy": f"{model_bundle['accuracy'] * 100:.1f}%",
#     }

@app.get('/health')
def health():
    return{
        'status': 'healthy',
        'model_status': 'loaded',
        'model_version': model_bundle['version'],
    }

#Endpoint(Get< post)

@app.post('/predict') 
def predict(data: WineInput):
    features = np.array([ 
        [
            data.alcohol, 
            data.malic_acid,
            data.ash,
            data.alcalinity_of_ash,
            data.magnesium,
            data.total_phenols,
            data.flavanoids,
            data.nonflavanoid_phenols,
            data.proanthocyanins,
            data.color_intensity,
            data.hue,
            data.diluted_wines,
            data.proline,
        ]
    ])
    
    predicted_class = int(model.predict(features)[0]) 
    probabilities = model.predict_proba(features)[0] 
    confidence = float(probabilities[predicted_class]) 

    return{
            'predicted_class': predicted_class,
            'predicted_label': target_names[predicted_class],
            'confidence': round(confidence *100,2),
            'model_version': model_bundle['version'],
}
