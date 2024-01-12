# 1. Exploratory Data Analysis

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

# 2. Data and Sampling Distribution

Random selection of data can reduce biase and yield a higher quality dataset than would result from just using the conveniently available data. Knowledge of various sampling and data-generating distributions allows us to quantify potential erros in an estimate that might be due to random variation. At the same time, the bootstrap (sampling with replacement from an observed dataset) is an attractive "one size its all" method to determin possible error in sample estimates

## Random Sampling
- Sampling = A subset from a larger dataset
- Population = The larger dataset or idea of a dataset
- N = the size of the population (sample)
- Random sampling = Drawing elements into a sample at random.
- Stratified sampling = Dividing the population into strata and randomly sampling from each strata.
- Stratum (strata) = A homogeneous subgroup of a population with common characteristics.
- Simple random sample = the sample that reuslts from randdom sampling without stratifying the population.
- Bias = Systematic error.
- Sample bias = A sample that misrepresents the population

## Size vs. Quality: When Does Size Matter?
- Small amount of randomly sampled data are better. So when are massive amounts of data needed? When there is a problem wehre a more data accumuated, the better the results.

## Sample Mean vs. Population Mean
- Even in the era of big data, random sampling remains an imporant arrow in the data scientist's quiver.
- Bias occurs when measurements or observations are systematically in error because they are not representative of the full population.
- Data quality is often more important than data quantity, and random sampling can reduce bias and facilitate quality improvement that would otherwise be prohibitively expensive.

## Selection Bias
- Selection bias = Bias resulting from a way in which observations are selected
- Data snooping = Extensive hunting through data in search of something interesting
- Vast search effect = Bias or nonreproducibility resulting from repeating data modeling, or modeling data with large numbers of predictor variables.
- Specifying a hypothesis and then collecitng data following randomization and random sampling principles ensures against bias.
- All other forms of data analysis run the risk of bias resulting from the data collection/analysis process (repeated running of models in data mining, data snooping in research, and after-the-fact selection of interesting events).

## Sampling Distribution of a Statistic
- Sample statistic = A metric calculated from a sample of data drawn from a larger population.
- Data distribution = the frequency distribution o individual values in a dataset.
- Sampling distribution = the freqquency distirbution of a sample statistic over many samples or resamples
- Central Limit Theorem = the tendency of the sampling distribution to take a normal shape as sample size rises.
- Standard error = the variability (standard deviation) of a sample statistic over many samples (not to be confused with standard deviation), which by itself, refers to variability of individual data values.

## Bootstrap
- Bootstrap sample = a sample taken with replacement from an observed dataset
- Resampling = the process of taking repeated samples from observed data; includes both bootstrap and permutation (shuffling procedures). 

## Resampling vs Bootstrapping
- The bootstrap (sampling with replacement from a dataset) is a powerful tool for assessing the variability of a sample statistic.
- The bootstrap can be applied in similar fashion in a wide variety of circumstances, without extensive study of mathematical approximations to sampling distributions.
- It also allows us to estimate sampling distributions for statistics where no mathematical approximation has been developed.
- When applied to predictive models, aggregating multiple bootstrap sample predictions (bagging) outperforms the usee of a single model.

## Confidence Intervals
- Confidence level = the percentage of confidence intervals, constructed in the same way from the sample population, that are expected to contain the statistic of interest.
- Interval endpoints = The top and bottom of the confidence interval.
- Confidence intervals are the typical way to present estimates as an interval range.
- The more data you have, the less variable a sample estimate will be.
- The lower the level of confidence you can tolerate, the narrower the confideence interval will be.
- The bootstrap is an effective way to construct confidence intervals.

## Normal Distribution
- Error = the difference between a data point and a predicted or average value.
- Standardize = subtract the mean and divide by the standard deviation.
- Z-score = the result of standardizing an individual data point.
- Standard normal = A normal distribution with mean = 0 and standard deviation = 1
- QQ-Plot = A plot to visualize how close a sample distribution is to a specified distribution, aka, normal distribution.
- Normal distribution was essential to the historical development of statistics, as it permitted mathematical approximation of uncertainty and variability.
- While raw data is typically not normally distributed, errors often are, as are averages and totals in large samples.
- To convert data to z-scores, you subtract the mean of the data and divide by the standard deviation; you can then compare the data to a normal distribution.
- Most data is not normally distributed.
- Assuming a normal distribution can lead to underestimation of extreme events (black-swans)

## Long-Tailed Distributions
- Tail = A long narrow portion of frequency distribution, where relatively extreme values occur at low frequency.
- Skew = Where one tail of a distribution is longer than the other.

## Student's t-Distribution
- The t-distribution is a normally shaped distribution, except that it is a bit thicker and longer on the tails. It is used extensively in depicting distributions of sample statistics.
- The t-distribution is acutally a family of distributions resembling the normal distribution but with thicker tails.
- The t-distribution is widely used as a reference basis for the distribution of sample means, differences between two sample means, regression parameters, and more.

## Binomial Distribution
- Yes/no (binomial) outcomes lie at the heart of analytics since they are often the culmination of a decision or other process; buy/don't buy, click/don't click,... Central to understanding the binomial distribution is the idea of a set of trials, each trial having two possible outcomes with definite probabilities.
- Trial = An event with a discrete outcome (coin flip)
- Success = The outcome of interest for a trial
- Binomial = having two outcomes (yes/no).
- Binomial trial = A trial with two outcomes 
  - Bernoulli trial.
- Binomial distribution = distribution of number of successes in x trials
  - Bernoulli distribution.
- Binomial outcomes are important to model, since they represent, among other things, fundamental decisions (buy or don't buy, click or don't click)
- A binomial trial is an experimentt with two possible outcomes: one with probability p and the other with probability 1 - p.
- With large n, and provided p is not too close to 0 or 1, the binomial distribution can be approximated by the normal distribution

## Chi-Square Distribution
- An important idea in statistic is departure from expectation, especially with respect to category counts. Expectation is defined loosely as "nothing unusual or o note in the data" (eg, no correlation between variables or predictable patterns). This is also termed as "null hypothesis".
- The chi-square distribution is typically concerned with counts of subjects or iterms falling into categories.
- The chi-square statistic measures the extent of departure from what you would expect in a null model.

## F-Distribution
- The F-distribution is used with experiments and linear models involving measured data.
- The F-statistic compares variation due to factors of interest to overall variation

## Poisson and Related Distribution
- Many processes produce events randomly at a given overall rate - visitors arriving at a website, or cars arriving at a toll plaza (events spread over time); imperfections in a square meter of fabric, or typos per 100 lines of code (events spread over space)
- Lambda = the rate (per unit of time or space) at which events occur.
- Poisson distribution = the frequency distribution of the number of events in sampled units of time or space.
- Exponential distribution = the frequency distribution of the time or distance from one event to the next event.
- Weibull distribution = a generalized version of the exponential distribution in which the event rate is allowed to shift over time

## Poisson Distributions
- Poisson distribution tell us the distribution of events per unit of time or spacewhen we sample many such units. It is useful when addressing queuing questions such as "How much capacity do we need to be 95% sure of fully processing the internet traffic that arrives on a server in any five-second period?"

## Weibull Distribution
- In manay cases, the event rate does not remain constatn over time. If the period over which it changes is much longer than the typical interval between evetns, there is no problem; you just subdivide the analysis into the segments where rates are relatively constatn. If, however, the event rate changes over the time of the interval, the exponential (or Poisson) distributions are no longer useful. This is likely to be the case in mechanical failure - the risk of failure increases as time goes by. The Weibull distribution is an extension of the exponential distirbution in which the event rate is allowed to change, as specified by a shape parameter.
- Because the Weibull distributionn is used with time-to-failure analysis instead of event rate, the second parameter is expressed in terms of characteristic life, rather than in terms of ratte of events per interval

## Poisson & Weibull
- For evetns that occur at a constannt rate, the number of events per unit of time or space can be modeled as Poisson distributionn.
- You can also model the time or distance between one event and the next as an exponential distribution
- A changing event rate over time (ie, an increase probability of device failure) can be modeled with the Weibull distribution.