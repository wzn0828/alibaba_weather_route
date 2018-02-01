# 3D A Star with Reinforcement Learning

This repo is used for "Future challenge Helping Balloons Navigate the Weather"
([English site](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100067.5678.1.3d16c911DB1wX4&raceId=231622&_lang=en_US)
, [Chinese site](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.59d64078pngYE6&raceId=231622))

### Key words 

3D A star, A*, tabular Reinforcement Learning, Dyna-Q, Double Q-learning, Double Expected Sarsa


## Getting Started
(1) Downloading the data and unzip into .csv files.

(2) Modify the path in the file `/config/diwu.py` (or the file of your choice):
set the data root directory `dataroot_dir=''` as where you have put them. 
Name the corresponding file, e.g. `TestForecastFile=''`

Run the `plt_forecast_wind_test_multiprocessing` logic by setting all data logic in `/config/diwu.py` to `False` and
 `plt_forecast_wind_test_multiprocessing` to `True`. The logic is executed by
```bash
python main.py
```

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
