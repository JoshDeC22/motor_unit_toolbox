# Motor Unit Toolbox

## Overview 
This repository contains functions to analyse motor unit (MU) behaviour, from computing basic firing and motor unit action potential (MUAP) properties, to comparing sets of spike trains and tracking MUAPs.

## Table of Contents
- [Installation](#installation)
- [Quick start](#quickstart)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Installation 
This toolbox is installable via pip with:
```sh
pip install git+https://github.com/imendezguerra/motor_unit_toolbox
```

To install it in editable mode, please clone the repository in your project's folder and run:
```sh
pip install -e ./motor_unit_toolbox
```
Once the toolbox has been installed you can just import the corresponding packages as:
```python
from motor_unit_toolbox import spike_comp
```
#### Prerequisites
When installing the package, `pip` will automatically install the required packages stored in `requirements.txt`. 

If you decide to clone the repository, then you can replicate the environment with:
```
conda env create -f environment.yml
```
The file was constructed without the build so it should be compatible with Os, Windows, and Linux.

#### Local setup guide
To set up the project locally do the following:

1. Clone the repository:
    ```sh
    git clone https://github.com/imendezguerra/motor_unit_toolbox.git
    ```
2. Navigate to the project directory:
    ```sh
    cd motor_unit_toolbox
    ```
3. Create the conda environment from the `environment.yml` file:
    ```sh
    conda env create -f environment.yml
    ```
4. Activate the environment:
    ```sh
    conda activate motor_unit_toolbox
    ```

## Quick start 
The package is composed of the following modules:
- `muap_comp.py`: Functions to compare and track MUAPs.
- `spike_comp.py`: Functions to compare spike trains between paired or unpaired sets, as well as within sets. Main metrics are rate of agreement, precision, sensitivity, F1 score, true positives, false positives, and false negatives.
- `props.py`: Functions to extract MU properties such as discharge rate, pulse to noise ratio, silhouette measure, and coefficient of variation of the interspike intervals, as well as MUAP features.
- `plots.py`: Functions to plot the spike trains, MUAPs, and grouped MUAPs.


## Contributing
We welcome contributions! Here’s how you can contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/newfeature`).
3. Commit your changes (`git commit -m 'Add some newfeature'`).
4. Push to the branch (`git push origin feature/newfeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite this repository:

```sh
@software{Mendez_Guerra_Motor_Unit_Toolbox,
author = {Mendez Guerra, Irene},
title = {{Motor Unit Toolbox}},
url = {https://github.com/imendezguerra/motor_unit_toolbox},
version = {1.0}
}
```
## Contact

For any questions or inquiries, please contact us at:
```sh
Irene Mendez Guerra
irene.mendez17@imperial.ac.uk
```