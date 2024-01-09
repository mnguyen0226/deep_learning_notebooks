# Exploratory Data Analysis

## Key Terms for Estimates of Location
- Mean = The sum of all values divided by the number of values.
- Weighted Mean = The sum of all values times a weight divided by the sum of the weights.
- Median = The value such that one-half of the data lies above and below.
- Percentile = The value such that P percent of the data lies below.
- Weighted Median = The value such that one-half of the sum of the weights lies above and below the sorted data.
- Trimmed Mean = The average of all values after dropping a fixed number of extreme values.
- Robust = Not sensitive to extreme values.
- Outlier = A data value that is very different from most of the data.

## Why use weighted mean?
- Some values are intrinsically more variable than others, and highly variable observations are given a lower weight. For example, if we are taking the average from multiple sensors and one of the sensors is less accurate, then we might downweight the data from that sensor.
- The data collected does not equally represent the different groups that we are interested in measuring. For example, because of the way an online experiment was conducted, we may not have a set of data that accurately reflects all groups in the user base. To correct that, we can give a higher weight to the values from the groups that were underrepresented.

## Variability Metrics
- Deviation = Difference between the observed values and the estimate of location.
  - Error, Residual
- Variance = the sum of squared deviations from the mean divided by n - 1 where n is the number of data values.
  - Mean-Squared-Error
- Standard deviation = the square root of the variance
- Mean absolute deviation = the mean of the absolute values of the deviations of from the mean
  - L1-Norm, Manhattan Norm
- Median absolute deviation from the median = the median of the absolute values of the deviations from the median.
- Range = the difference between the largest and the smallest value in the dataset
- Order Statistic = Metrics based on the data values sorted from smallest to biggest
  - Ranks
- Percentile = The value such taht P percent of the values take on this value or less and (100 - P) percent take on this value or more.
  - Quantile
- Interquartile range = The difference between the 75th percentile and the 25th percentile
    - IQR

## Exploring the Data Distribution
- Boxplot = Quick way to visualize the distribution of data 
  - Box and whiskers plot
- Frequency Table = A tally of count of numeric data values that fall into a set of intervals (bins)
- Histogram = A plot of frequency table with the bins on the x-axis and the count on the y-axis. Bar charts should not be confused with histogram
- Density plot = a smoothed version of the histogram, often based on the kernel density estimate

### Key Ideas
- A frequency historgram plots frequency counts on the y-axis and variable vvalues on the x-axis, it gives a sense of the distribution of the data at a glance.
- A frequency table is tabular version of the frequency counts found in a histogram.
- A boxplot - with the top and bottom of the box at the 75th and 25th percentiles respectively - also gives a quick sense off the distribution of the data; it is often used in side-by-side displays to compare distributions.
- A density plot is a smoothed version of a histogram; it requires a function to estimate a plot based on the data.

## Exploring Binary and Categorical Data
- Mode = the most commonly occurring category or value in a dataset.
- Expected value = When the categories can be associated with a numeric value, this gives an average value based on a category's probability of occurence.
  - Expected value is really a form of weighted mean: it adds the ideas of future expectations and probability weights, often based on subjective judgement. Expected value is a fundamental concept in business valuation and capital budgeting, for example, the expected value of 5 years off profits from the new acquisition, or the expected cost savings from new patient management software at a clinic.
- Bar charts = the frequency or propertion for each category plotted as bars.
- Pie charts = the frequency or proportion for each category plotted as wedges in a pies

## Exploring Two or More Varaibles (bivariate or multivariate analysis)
- Contingency table = A tally of counts between two or more categorical variables.
- Hexagonal binning = A plot of two numeric variables with the records binned into hexagons.
- Contour plot = A plot showing the density of 2 numeric variables like a topographical maps.
- Violin plot = Similar to a boxplot but showing the density estimate