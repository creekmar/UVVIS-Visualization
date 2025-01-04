# UVVIS Visualization for Lab Automation
A way to visualize the T80 absorption data for the OPV lab automation project described [here](https://github.com/changhwang/Lab_Automation/tree/master?tab=readme-ov-file#further-details).

## Description
Reads through all csv files containing T80 spectral data in a folder. The first column of the dataframe after the index should be Wavelength. 
The rest of the columns will be the absorbance value at the specified timestamp. The data is visualized through the method described in Figure S10 of the article [here](https://doi.org/10.1126/science.adi1407).

![UVVIS Visualization](Images/uvvis_visual.png)

1. The code will plot a line graph of Absorbance vs. Wavelength, where time is represented as the color of the plot.
The time increases from dark to lighter color.

2. From the previous graph, a spectral deconvolution is performed on the dataset. The pristine component and degraded component are shown on a graph.

3. The concentration of the spectral deconvolution is normalized through the natural log. At least 5 data points are used to fit a linear regression. A data point is added each time to fit a new regression.
The best regression is kept. 

## Getting Started
### Requirements
This package was developed using Python 3.10. 
The required packages are listed below.

- NumPy
- Matplotlib
- Pandas
- pyMCR

## Notice
The code was partially taken from the journal article's main code repository [here](https://github.com/RBCanty/MIT_AMD_Platform). 
The code can be found by going to the file Shell_SP, then the project_processing.py, then the function _advancedregression.
Due to shallow understanding of spectral deconvolution the written code in this repository may or may not be flawed. 
You can run this file alone to get an initial 30 points generated or points generated from a file. The code will use umap and pca reduction to plot the points on a graph to see the spread of the data within the total sample space.

### scikit_plot
A file with some functions to visualize data, including graphing 3d true objective, 4d data, pca, umap, optimization trace, and printing scikit-optimization results. This is utilized when visualizing initial sample space points and understanding scikit bayesian optimization results.



