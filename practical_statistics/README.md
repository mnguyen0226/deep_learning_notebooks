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

# 3. Statistical Experiments and Significance Testing
- Design of experiments is a cornerstone of the practice o statistics, with application in virtually all areas of research. The goal is to design an experiment in order to confirm or reject a hypothesis.
- Data scientists often need to conduct continual experiments, particularly regarding user interface and product marketing. 
- Inference = apply the experiment results, which involve a limted set of data, to a larger process or population. Here is the classifcal statistical inference pipeline
  - Step 1. Formulate hypothesis.
  - Step 2. Design experiment.
  - Step 3. Collect data.
  - Step 4. Inference / Conclusion.

## A/B Testing
- An A/B test is an experiment with two groups to establish which of two treatments, products, procedures, or the like is superior. Of ten one of the two treatments is the standard existing treatment, or no treatmentt. If a standard (or no) treatement is used, it is called the control. A typical hypothesis is that a new treatment is better than the control.
- Terms:
  - Treatment = something (drug, price, web headline) to which a subject is exposed.
  - Treatment group = a group of subjects exposed to a specific treatment.
  - Control group = a group of subject exposed to no (or standard) treatment.
  - Randomization = the process o randomly assigning subjects to treatments.
  - Subjects = the items (web vistors, patients,...) that are exposed to treatments.
  - Test statistic = the metric used to measure the efect of the treatment.

- A/B tests are common in web design and marketing, since results are so readily measured. For example:
  - Testing two soil treatments to determin which produces better seed germination
  - Test two price to determine which yield more net profit.
  - Test two web headlines to determine which produces more clicks.
  - Test two web ads to determine which generates more conversions.

- A proper A/B test has subjects that can be assigned to one treatment or another. The subject might be a person, a plant seed, or a web visitor; the key is that the subject is exposed to the treatment. Ideally, subjects are randomized (assigned randomly) to treatments. In this way, you know that any difference between the treatment groups is due to one of two things:
  - The effect of the different treatments.
  - Luck of draw in which subjects are assigned to which treatments (ie, the random assignment may be resulted in the naturally better-performing subjects being concentrated in A or B)

## Why Have a Control Group?
- Why not skip the control group and just run an experiment applying the treatment of interest to only one group, and compare the outcome to prior experience?
- Well, without a control group, there is no assurance that "all other things are equal" and that any difference is readlly due to the treatment (or to chance). When you have a control group, it is subject to the same conditionals (except for the treatment of interest) as the treatment group. If you simply make a comparison to "baseline" or prior experience, other factors besides the treatment, might differ.
- A/B testing in data science is typically used in a web context. Treatments might be the design of a web page, the price of a product, the wording of a headline. Some thought is required to preserve the principles of randomization. Typically the subject in the experiment is a web visitor, the outcomes we are interested in measuring are clicks, purchases, visit duration, number of pages visited, weather a particular page is visited and the like. Multiple behaviour metrics might be collected and be of interest, but if the experiment is expected to lead to a decision between treatment A and treatment B, a single metric, or test statistic, needs to be established beforehand. Selecting a test statistic after the experiment is conducted opens the door to researcher bias.

## Why Just A/B? Why Not C, D,...?
- A/B tests are popular in the marketing and ecommerce worlds, but are far from the only type of statistical experiment. Additional treatments can be included.Subjects might have repeated measurements taken. 
- Data scientists are less interested in the question: "Is the diference between price A and price B statistically significantt?" than in the question: "Which, out of multiple possible prices, is best?"
- In summary, subjects are assigned to 2 or more groups that are treated exactly alike, except that the treatment under study differ from one group to another.

## Hypothesis Tests
- Hypothesis tests (significance tests) are ubiquitous in the traditional statistical analysis of published research. Their purpose is to help you learn whether random chance might be responsible for an observed effect.
- Terms:
  - Null hypothesis = the hypothesis that chance is to blame.
  - Alternative hypothesis = counterpoint to null (what you hope to prove)
  - One-way test = hypothesis test that counts chance results only in one direction.
  - Two-way test = hypothesis test that counts chance results in two directions.
- An A/B test is typically constructed with a hypothesis in mind. 
  - For instance, a hypothesis might be that price B produces higher profit.
- Why do we need a hypothesis? Why not just look at the outcome of the experiment and go with whichever treatment does better?
- The answer lies in the tensdency of the human mind to underestimate the scope of the natural random behaviour.

## Misinterpreting Randomness
- You can observe the human tendency to underestimate randomness in this experiment. Ask several friends to invent a series of 50 coin flips: have them write down a series of random Hs and Ts. Then ask them to actually flip a coin 50 times and write down a results. Have them put the real coin flip results in one pile, and the made-up results in another. It is easy to tell which results are real: The real ones will have the longer runs of Hs or Ts. In a set of 50 real coin flips, it is not at all unusal to see 5 or 6 Hs or Ts in a row. However, when most o us  are inventing random coin flips and we have gotten 3 or 4 Hs in a row, we tell ourselves that, for the series to look random, we had better switch to T.
- The other side oo this coin, so to speak, is that when we do see the real-world equivalent of 6 Hs in a row (e.g, when one headline outperforms another by 10%), we are inclined to attribute it to something real, not just to chance.
- In a properly designed A/B test, you collect data on treatments A and B in such a way that any observed diference between A and B must be due to either:
  - Random chance in assignment of subjects.
  - A true difference between A and B.

## The Null Hypothesis
- Hypothesis tests use the following logic: "Given the human tendency to react to unusual but random behaviour and interpret it as something meaningful and real, in our experiments, we will require prood that the difference between groups is more extreme than what chance might reasonably produce". This involves a baseline assumption that the treatments are equivalent, and any difference between the groups is due to chance. This baseline assumption is termed the null hypothesis. Our hope, then, is that we can in fact prove the null hypothesis wrong and show that the outcomes for groups A and B are more diffent than what chance might produce.
- A null hypothesis is a logical construct embodying the notion that nothing special has happened, and any effect you observe is due to random chance.
- The hypothesis test assumes that the null hypothesis is true, creates a "null model" (a probability model), and tests whether the efect you observe is a reasonable outcome of that model.

## Reampling
- Resampling in statistics means to repeately sample values from observed data, with a general goal of assessing random variability in a statistic. It can also be used to assess and improve the accuracy of some ML model.
- 2 types of resampling procedures: bootstrap and permutation test.
- Permutation test = the procedure of combining 2 or more samples together and randomly (or exhaustively) reallocating the observations to resamples.
  - Randomization test, random permutation test, exact test.
- Resampling = Drawing additional samples ("resamples") from an observed dataset.
- With or without replacement = in sampling, whether or not an item is returned to the sample before the next draw


## Permutation Test
- Two or more samples are involved, typically the groups in an A/B or other hypothesis test. We test by randomly drawing groups from this combined set and seeing how much they differ from one another.
- Steps:
  - 1. Combine the results from the diferent groups into a single dataset.
  - 2. Shuffle the combined data and then randomly draw (without replacement) a resample of the sample size as group A (clearly it will contain some data from the other groups).
  - 3. From the remaining data, randomly draw (without replacement) a resample of the same size as group B.
  - 4. Do the same for groups C, D. You have now collected one set of resamples that mirror the sizes of the original samples.
  - 5. Whatever statistic or estimate was calculated for the original samples (e.g., difference in group proportions), calculate it now for the resamples, and record; this constitutes one permutation iteration.
  - 6. Repeat the previous steps R times to yield a permutation distributeion of the test statistic

## Web Stickiness Example
- A company selling a relatively high-value service wants to test which of two web presentations does a better selling job. Due to the high value of the service being sold, sales are infrequent and the sales cycle is lengthy; it would take too long to accumulate enough sales to know which presentation is superior. So the company decides to measure the results with a proxy variable, using the detailed interior page that describes the service.

## Exhaustive and Bootsrap Permutation Tests
- In addition to the proceding random shuffling procedure (random permutation test or randomization test), there are two variants of the permutation test:
  - An exhaustive permutation test.
  - A bootstrap permutation tes.

## Permutation Tests: The Bottom Line for Data Science
- In a permutation test, multiple samples are combined and then shuffled.
- The shuffled values are then divided into resamples, and the statistic of interest is calculated.
- This process is then repeated, and the resampled statistic is tabulated.
- Comparing the observed value of the statistic to the resampled distribution allows you to judge whether an abserved difference between samples might occur by chance.

## Statistical Significance and p-Values
- Statistical significance is how statisticians measure whether an experiment (or even a study of existing data) yields a more result more extreme than what chance might produce. If the result is beyong the realm of of chance variation, it is said to be statistically significant.
- Terms
  - p-value = Given a chance model that embodies the null hypothesis, the p-value is the probability of obtaining results as unusual or extreme as the observed results.
  - Alpha = the probability threshold of "unusualness" that chance results must surpass for actual outcomes to be deemed statistically significant.
  - Type 1 error = Mistakenly concluding an eect is real (when it is due to chance).
  - Type 2 error = Mistakenly concluding an effect is due to chance (when it is real).
- Significance tests are used to determine whether an observed effect is within the range of chance variation for a null hypothesis model.
- The p-value is the probability that results are extreme as the observed results might occur, given a null hypothesis model.
- The alpha value is the threshold of "unusualness" in a null hypothesis chance model.
- Significance testing has been much more relevant for formal reporting of research than for data science (but has been fading recently, even for the former).