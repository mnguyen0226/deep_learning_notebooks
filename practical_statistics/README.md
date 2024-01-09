# EDA

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