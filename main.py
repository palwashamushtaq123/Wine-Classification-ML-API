from pathlib import Path #Handle Files aur folders's paths.

import joblib #save tained Ml models
import numpy as np 
from fastapi import FastAPI, HTTPException, Request #use to make web application,HTTPException use to send error,
from fastapi.responses import HTMLResponse #Browser ko HTML page bhejna.Agar ye na ho to browser JSON show karega.
from fastapi.staticfiles import StaticFiles #save css,Images,JavaScript
from fastapi.templating import Jinja2Templates # to render HTML template,and send python variable in html
from pydantic import BaseModel, Field #pydantic use to validate user input either user write right or wrong

BASE_DIR = Path(__file__).resolve().parent #it is current floder project address
print(f'BASE DIR:{BASE_DIR}')
model_path = BASE_DIR/'model.pkl' 

if not model_path.exists():
    raise RuntimeError("model.pkl was not found.Run 'python train.ipynb' first.")

model_bundle = joblib.load(model_path) # it open saved model.pkl
model = model_bundle['model'] #getout ml model
target_names = model_bundle['target_names'] #get lable here target lables are setosa,..
accuracy = model_bundle['accuracy'] #get model save training accurary
version = model_bundle['version'] # get model version

print(type(model_bundle["accuracy"]))
print(model_bundle["accuracy"])

#application create => app = FastAPI()
app = FastAPI(
    title= 'Wine Classification Web App', #create aplication title
    version = model_bundle['version'], #show saved version
)

#This step do to make accessible Css, Java Script
app.mount(
    '/static',
    StaticFiles(directory=BASE_DIR/'static'),
    name= 'static',
)

# This tells HTML files kis folder mein hain => templates = Jinja2Templates(...) 
templates = Jinja2Templates(directory= BASE_DIR/'templates')

# This APL input structure. user have to give 4 input values
class WineInput(BaseModel):
    alcohol: float = Field(gt=0) # gt=0 means greater than 0.
    malic_acid: float = Field(gt=0) #Agar user 0 ya negative number 
    ash: float = Field(gt=0) #dega to FastAPI validation error return karega.
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

#Jab browser kholta hai: to y funcation run hota hi throught this  => http://127.0.0.1:8000
# => templates.TemplateResponse(...) Ye index.html browser ko bhejta hai.
# Saath hi context ke through data bhi bhejta hai:
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
# this step run when we check our API without css,html 
# one get('/') or can say this home function will use 
#if we use in css and hlml we must have unactive it. double home wont work
# @app.get('/')
# def home():
#     return{
#         "message": "Welcome to the Wine Classification API",
#         "model_version": model_bundle["version"],
#         "model_accuracy": f"{model_bundle['accuracy'] * 100:.1f}%",
#     }

#e check karta hai ke API aur model sahi chal rahe hain ya nahi.
@app.get('/health')
def health():
    return{
        'status': 'healthy',
        'model_status': 'loaded',
        'model_version': model_bundle['version'],
    }

#Endpoint(Get< post)

@app.post('/predict') # this main endpoint of the project
def predict(data: WineInput):
    features = np.array([ #Model ko input 2D array mein diya jata hai.
        [
            data.alcohol, #Outer [] = samples, Inner [] = features
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
    
    predicted_class = int(model.predict(features)[0])  # Model prediction karta hai.
    probabilities = model.predict_proba(features)[0] #ye har class ki probability return karta hai.
    confidence = float(probabilities[predicted_class]) #Predicted class ki probability nikalta hai.

    return{
            'predicted_class': predicted_class,
            'predicted_label': target_names[predicted_class],
            'confidence': round(confidence *100,2),
            'model_version': model_bundle['version'],
}





#predict(WineInput(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2))
