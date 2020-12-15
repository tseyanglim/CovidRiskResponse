# CovidRiskResponse
Code & model files for Rahmandad & Lim (2020), Risk-driven responses to COVID-19 eliminate the tradeoff between lives and livelihoods ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3747254))

For any questions please contact [Tse Yang Lim](mailto:tylim@mit.edu)

### Analysis Code
Contains the Python code used for data pre-processing and model estimation, in .ipynb and .py formats.

**Important:** The model estimation code is intended to work with an experimental parallelised Vensim engine. With appropriate modifications to the main function calls (but not the analytical procedure), the same analysis can be run on regular commercially available Vensim DSS, though it will take *much* longer. Please contact [Tom Fiddaman](mailto:tom@ventanasystems.com) for information on the experimental Vensim engine.

#### Graphing Code
Contains the Python code used for producing all figures and summary statistics used in the paper and supplement, in .ipynb and .py formats

### Data
Contains Vensim data files (.vdf) used in model estimation, the input .tab file processed from [OurWorldInData](https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv) based on the code in `Analysis Code`, and the .xlsx file used to calculate country-specific IFRs based on age distributions (see S2 of the Supplement for explanation).

### Results
Contains summary output .tab files from the model estimation presented in the paper and sensitivity analyses in the supplement, as well as .jpg versions of all figures used, which can be reproduced using the code in `Graphing Code`. Includes full country-by-country parameter estimates accompanying Supplement S3.

### Vensim Files
Contains the main Vensim model file (.mdl) and other supplementary Vensim files used for model estimation (e.g. optimization control, payoff definition, savelist files, and so on), as well as sub-model used for illustrative parameter sensitivity analysis.