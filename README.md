# BioCredits Calc

This repository contains the necessary files for calculating BioCredits. The main computation is demonstrated in the `testing_notebook.ipynb` Jupyter notebook.

## Getting Started

These instructions will guide you through setting up your environment and running the notebook.

### Prerequisites

Ensure you have [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system. These tools provide an easy way to manage environments and packages for your projects.

### Setting Up Your Environment
```
sudo apt-get update
sudo apt-get install git
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
~/miniconda3/bin/conda init
source ~/.bashrc
sudo apt-get install gdal-bin
sudo apt install ffmpeg
```

1. **Clone the Repository**

   First, clone this repository to your local machine using Git.

   ```bash
   git clone https://github.com/savimbo/biocredits-calc.git
   cd biocredits-calc
   ```
   
### Setting Up Your Environment


2. **Create a Conda Environment**

Create a new Conda environment named `biocredits_env` with Python 3.10. Replace `biocredits_env` with a name of your choice if desired.
```bash
conda create --name biocredits_env python=3.10
```

3. **Activate the Environment**

Activate the newly created environment.
```bash
conda activate biocredits_env
```

4. **Install Requirements**

Install the required packages listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

### Running the Notebook

With the environment set up and activated, you can now run the Jupyter notebook.

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open `testing_notebook.ipynb`**

Navigate through the Jupyter Notebook interface in your browser to the location of the cloned repository. Open the `testing_notebook.ipynb` notebook.

3. **Run the Notebook**


Execute the cells in the notebook to perform the calculations or analyses contained within.

## Contributing

Feel free to fork the repository and submit pull requests with enhancements or fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



