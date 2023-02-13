# LINE
## HOW TO USE
```
python3 line.py -order=1
```
in default, a file "model_line.txt" will be generated. Here are some useful parameter as followed:
- "-order": Set 1 for first order, 2 for second order
- "-dimensions": Set dimensions
- "-save": Set save model file path
- "-sample_times": Set sample times
- "-negative_samples": Set negative samples
- "-alpha": Set alpha
- "-net": Set graph file path

## PREDICTION
```
python3 predict.py -model=model_line.txt -pre=recommend.dat
```
- "-model": Set using model file name 
- "-pre": Set prediction file name

## SCORE
```
python3 score_top_k.py -pre=recommend.dat
```
- "-pre": Set using prediction file name