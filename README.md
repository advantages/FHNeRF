# FHNeRF
The essential code for training FHNeRF
To produce the main results, just run the following command:
```
python main.py -ld logs_dir -fd --num_layers 5 --layer_size 20
python main.py -ld logs_dir -fd --num_layers 5 --layer_size 100
```
The --num_layers is not suggested to be larger than 10
