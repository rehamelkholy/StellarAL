# Active Learning for Stellar Spectral Classification

## Description
This repository contains the code associated with the paper "Optimized sampling of SDSS-IV MaStar spectra for stellar classification using supervised models". The project focuses on the application of active learning methodologies to enhance the efficiency and accuracy of classifying stellar spectra. Traditional machine learning approaches for spectral classification often require vast amounts of labeled data, which is labor-intensive to obtain. Active learning, by contrast, strategically selects the most informative samples for labeling, thus minimizing the required labeled dataset while maximizing model performance.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [License](#license)
- [Citing](#citing)
- [Contact](#contact)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/rehamelkholy/StellarAL.git
    ```
2. Navigate to the project directory:
    ```bash
    cd StellarAL
    ```
3. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
4. Activate the virtual environment:
    - On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
5. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
- To reproduce the results of the paper, you can run the `ipynb` files in the same order as in [Files](#files).
- The original SDSS data files used in this study can be downloaded from the following links:
    - [MaStar Spectra](https://data.sdss.org/sas/dr17/manga/spectro/mastar/v3_1_1/v1_7_7/mastar-goodspec-v3_1_1-v1_7_7.fits.gz)
    - [MaStar Parameters](https://data.sdss.org/sas/dr17/manga/spectro/mastar/v3_1_1/v1_7_7/vac/parameters/v2/mastar-goodstars-v3_1_1-v1_7_7-params-v2.fits)

## Files
###### Miscellaneous
- `README.md` : Documentation file
- `requirements.txt` : List of Python dependencies
- `utils.py` : Pre-defined functions collected in one script to be imported at the beginning of each `ipynb` file
###### Jupyter Notebooks
1. `sec2_data.ipynb` : Initial data preparation and exploration
2. `sec3_1_preprocessing.ipynb` : Data pre-processing before applying AL & ML methods
3. `sec3_rand_vs_modal.ipynb` : Testing different AL sampling strategies against a random-sampling baseline
4. `sec3_n_instances.ipynb` : Testing performance improvement with increasing number of instances using the highest-performing AL sampling strategy
5. `sec3_4_metrics.ipynb` : Plotting an illustration of the AUC for different example models

## License
This project is licensed under the GPL License.

## Citing
If you use this code in your projects, you can cite it as
```latex
@misc{elkholy2024,
      title={Optimized sampling of SDSS-IV MaStar spectra for stellar classification using supervised models}, 
      author={R. El-Kholy and Z. M. Hayman},
      year={2024},
      eprint={2406.18366},
      archivePrefix={arXiv},
      primaryClass={astro-ph.SR},
      url={https://arxiv.org/abs/2406.18366}, 
}
```

## Contact
Dr. Reham El-Kholy has a PhD from Cairo University where she works as a Lecturer of Astronomy. If you have any questions, requests, or suggestions, you can contact her at [relkholy@sci.cu.edu.eg](mailto:relkholy@sci.cu.edu.eg). We hope you will find this useful!