from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
# Load the model from the file
model = joblib.load('Mark_predictor.pkl')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"], 
    allow_headers=["*"], 
    allow_credentials=True,
)

# Define the FastAPI input model
class StudentData(BaseModel):
    Student_id : int
    Time_Spent_on_speach: str
    Cumulative_Average_Quiz_Score: float
    speach: str


@app.post("/predict/")
def predict_score(data: StudentData):
    # Convert input data to a DataFrame
    input_data = pd.DataFrame({
        'Time_Spent_on_speach': [data.Time_Spent_on_speach],
        'Cumulative_Average_Quiz_Score': [data.Cumulative_Average_Quiz_Score],
        'speach':[data.speach]
    })

    # Convert 'Time_Spent_Studying' to numeric
    input_data['Time_Spent_on_speach'] = input_data['Time_Spent_on_speach'].str.extract('(\d+)').astype(int)

    # Predict the exam score
    predicted_scores = model.predict(input_data)

    # Return the prediction as JSON
    return {"predicted_scores": round(predicted_scores[0],2)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
