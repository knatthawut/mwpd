# PMap: Pre-training Models for Product Matching

This repository includes the code of PMAP and the results used in Semantic Web Challenge@ISWC 2020 on Mining the Web of HTML-embedded Product Data (https://ir-ischool-uos.github.io/mwpd/) for the product matching task (Task 1).

## Installation 
Install the dependency libraries using:
```
pip install -r requirements.txt
```

## Data 
Please find the data at the challenge website: https://ir-ischool-uos.github.io/mwpd/

## Run the experiments
To run the experiments, use:
```
python main.py
```
or load the fine-tuned model use:
```
python main.py -m specific_exisiting_fine-tuned_model
```

## Reference
Natthawut Kertkeidkachorn and Ryutaro Ichise, PMap: Ensemble Pre-training Models for Product Matchings. Mining the Web of HTML-embedded Product Data on Semantic Web Challenge@ISWC 2020 (to appear)