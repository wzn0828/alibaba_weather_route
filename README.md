# 3D A Star with Reinforcement Learning

This repo is used for "Future challenge Helping Balloons Navigate the Weather"
([English site](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100067.5678.1.3d16c911DB1wX4&raceId=231622&_lang=en_US)
, [Chinese site](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.59d64078pngYE6&raceId=231622))

### Key words 

3D A star, A*, tabular Reinforcement Learning, Dyna-Q, Double Q-learning, Double Expected Sarsa


## Getting Started
Note: all logics are executed by modifying the configure file and the command:
```bash
python main.py
```

(1) Downloading the data and unzip into .csv files.

(2) Modify the path in the file `/config/diwu.py` (or the file of your choice):
set the data root directory `dataroot_dir=''` as where you have put them. 
Name the corresponding file, e.g. `TestForecastFile=''` and their corresponding `.npy` files saving location `wind_save_path=''` .

(3) Run the `plt_forecast_wind_test_multiprocessing` logic by setting all data logic in `/config/diwu.py` to `False` and
 `plt_forecast_wind_test_multiprocessing` to `True`. Note: set the `num_threads=10` (it depends upon your total memory, if you have more memory, you can set it higher to better utilise the multiprocessing power). It will take up to 3 hours to finish the data extraction process.

This logic will take the .csv file for the wind predictions from various models given and output the `.npy` file for individual model-day-hour.

(4) Run the `plot_all_wind` logic by setting all data logic in `/config/diwu.py` to `False` and `plot_all_wind=True`.
`plot_test_model` and `plot_train_model` should also be set to `True` accordingly. `fig_save_train_path` should be set the the path where you want to store the `.png` files.

This script will simply save the wind models in a single `.png` file for better visualisation and comparison between different wind predictions.

### 3D A * algorithm

Run the logic `A_star_search_3D_multiprocessing`, hyper-parameters are as follows:
* **model_number**: which wind model are run upon
* **conservative**: a linear conservative cost

This logic will generate files in the `Experiments` folder with information stamp and time stamp. For a single wind model, it will take up to 1-2 hours (The bottleneck is the CPU cores you have on your PC).

### Tabular Reinforcement Learning for model fusion
 
 Run the logic `reinforcement_learning_solution_multiprocessing`,  hyper-parameters are as follows:





## Authors

* **Di Wu** - [stevenwudi](http://stevenwudi.github.io)
* **Zhennan Wang**
