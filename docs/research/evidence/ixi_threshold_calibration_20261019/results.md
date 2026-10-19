# IXI train-calibrated relation thresholds

|Split|Policy|Threshold|Precision|Recall|F1|β0 MAE|β1 MAE|
|---|---|---:|---:|---:|---:|---:|---:|
|train|baseline|0.5000|0.2030|0.0697|0.1037|16.6758|0.9805|
|train|train_max_f1|0.3820|0.1557|0.3169|0.2088|2.1875|16.3555|
|train|train_preserve_precision|0.4977|0.2034|0.0734|0.1079|16.3945|1.0156|
|val|baseline|0.5000|0.1623|0.0477|0.0737|19.2891|1.0391|
|val|train_max_f1|0.3820|0.1226|0.2338|0.1608|2.3203|19.5234|
|val|train_preserve_precision|0.4977|0.1654|0.0511|0.0780|19.0234|1.0859|
