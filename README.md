# 18-ha-2010-pj
Code-Abgabe zum Projektseminar "Wettbewerb künstliche Intelligenz in der Medizin" WiSe 2021/2022 vom Team CORrelation.
Wir verwenden ein tiefes neuronales Netz, genauer ein Transformer-Netz bzw. den Encoder-Teil des Transformers.

## How it works

1. Vorverarbeitung der rohen Input-Daten in der Preprocessing-Pipeline
2. Füttern der vorverarbeiteten Daten an das Netz, um es zu trainieren oder - falls ein trainiertes Modell schon vorliegt - um es für Vorhersagen zu nutzen.

## How to run

Für predict bzw. predict_pretrained und score haben wir uns an die Skeleton-Vorgaben gehalten und unseren Code entsprechend eingebettet.
Um die Prozesse "Train" bzw. "Finetuning" zu implementieren haben wir analog zur Vorgabe (wie in predict_pretrained.py) ein eigenes Gerüst gebaut (train_pretrained.py) und den Code in train.py eingebettet. train_pretrained.py ruft also train.py auf.

## Preprocessing-Pipeline

Die EKG-Rohdaten liegen als beliebig lange, eindimensionale Zeitreihe vor. Unser Transformer-Encoder verwendet eine feste Input-Size, weshalb im ersten Schritt das Signal in - je nach Gesamtlänge des Arrays - unterschiedlich viele, der Input-Size entsprechend lange Subarrays aufgeteilt wird (Skalpell.py).
Anschließend werden eine Rauschreduzierung mit einem 'Butterworth'- und einem 'Non Local Mean'- Filter und eine Baseline-Angleichung mittels 'LOESS Curve Fitting' (Vgl.: https://www.nature.com/articles/s41597-020-0386-x.pdf).

## Modell
Transformer-Encoder mit vorgeschalteten Convolutional Layern und Positional Encoding und nachgeschaltetem Feed-Forward Teil