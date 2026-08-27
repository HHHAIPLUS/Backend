from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

@dataclass
class Fold:
    train: list[dict]
    test: list[dict]

def walk_forward(rows: list[dict], min_train: int=300, test_size: int=100, step: int=100):
    rows=sorted(rows,key=lambda r:r['observed_at'])
    folds=[]; end=min_train
    while end+test_size<=len(rows):
        folds.append(Fold(rows[:end],rows[end:end+test_size])); end += step
    return folds

def evaluate_predictions(predictions: list[tuple[int,int,float]]) -> dict:
    if not predictions: return {'trades':0,'accuracy':0.0,'balanced_accuracy':0.0,'avg_return':0.0}
    correct=sum(1 for y,p,_ in predictions if y==p)
    returns=[r for _,_,r in predictions]
    classes=sorted(set(y for y,_,_ in predictions))
    recalls=[]
    for c in classes:
        actual=[(y,p) for y,p,_ in predictions if y==c]
        recalls.append(sum(1 for y,p in actual if y==p)/len(actual) if actual else 0)
    return {'trades':len(predictions),'accuracy':correct/len(predictions),'balanced_accuracy':sum(recalls)/len(recalls),'avg_return':sum(returns)/len(returns)}
