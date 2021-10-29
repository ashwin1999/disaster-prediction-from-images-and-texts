from pydantic.main import BaseModel


class Sentence(BaseModel):
    text: str


class Sentences(BaseModel):
    texts: list[str]


class onePrediction(BaseModel):
    prediction: float
    pred_class: str


class multiPrediction(BaseModel):
    prediction: list[float]
    pred_class: list[str]
