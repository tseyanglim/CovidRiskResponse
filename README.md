# CovidRiskResponse
Code & model files for submission version of Lim et al. (2023), Why Similar COVID-19 Policies Resulted in Different Outcomes: a Global, Behavioral Perspective

For any questions please contact [Tse Yang Lim](mailto:tylim@mit.edu)

### Analysis Code
Contains the Python code used for data pre-processing and model estimation, as well as separate Python code for cultural constructs regressions.

**Important:** The model estimation code is intended to work with an experimental parallelised Vensim engine. With appropriate modifications to the main function calls (but not the analytical procedure), the same analysis can be run on regular commercially available Vensim DSS, though it will take *much* longer. Please contact [Tom Fiddaman](mailto:tom@ventanasystems.com) for information on the experimental Vensim engine.

#### Graphing Code
Contains the Python code used for producing most figures and summary statistics used in the paper and supplement, as well as R code for Figure 1 in the main paper.

### Data
Contains:
1. Vensim data files (`.vdf`) used in model estimation, 
2. the input `.tab` files processed from [OurWorldInData](https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv) based on the code in `Analysis Code`, 
3. the `.xlsx` file used to calculate country-specific IFRs based on age distributions (see S2 of the Supplement for explanation), 
4. an `.RData` file drawn from [Oxford CGRT](https://github.com/OxCGRT/covid-policy-tracker) used to produce Figure 1 in the main paper, and
5. a `.csv` file containing Hofstede's cultural construct values appended to model estimation results.

### Results
Contains summary output .tab files from the model estimation presented in the paper and sensitivity analyses in the supplement, as well as .jpg versions of all figures used, which can be reproduced using the code in `Graphing Code`. Includes full country-by-country parameter estimates accompanying Supplement S3.

### Vensim Files
Contains the main Vensim model file (.mdl) and other supplementary Vensim files used for model estimation (e.g. optimization control, payoff definition, savelist files, and so on), as well as sub-model used for illustrative parameter sensitivity analysis.