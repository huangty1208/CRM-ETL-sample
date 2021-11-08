df = (spark
  .read                                              
  .option("inferSchema","true")                 
  .option("header","true")                           
  .csv("/FileStore/tables/train.csv"))

df.select('var_0').describe().show()


# quartile

def describe_pd(df_in, columns, deciles=False):

    if deciles:
        percentiles = np.array(range(0, 110, 10))
    else:
        percentiles = [25, 50, 75]

    percs = np.transpose([np.percentile(df_in.select(x).collect(), percentiles) for x in columns])
    percs = pd.DataFrame(percs, columns=columns)
    percs['summary'] = [str(p) + '%' for p in percentiles]

    spark_describe = df_in.describe().toPandas()
    new_df = pd.concat([spark_describe, percs],ignore_index=True)
    new_df = new_df.round(2)
    return new_df[['summary'] + columns]
  
describe_pd(df,num_cols) 
describe_pd(df,num_cols,deciles=True)


# Skewness and Kurtosis
from pyspark.sql.functions import col, skewness, kurtosis
df.select(skewness(var),kurtosis(var)).show()


# where
df.where(col("var_0").isNull()).count()
# filter
df.filter(col("var_0").isNull()).count()

# bins
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import udf
import numpy as np

var = "var_0"
# create the split list ranging from 0 to  21, interval of 0.5
split_list = [float(i) for i in np.arange(0,21,0.5)]
# initialize buketizer
bucketizer = Bucketizer(splits=split_list,inputCol=var, outputCol="buckets")
# transform
df_buck = bucketizer.setHandleInvalid("keep").transform(df.select(var).dropna())

# the "buckets" column gives the bucket rank, not the acctual bucket value(range), 
# use dictionary to match bucket rank and bucket value
bucket_names = dict(zip([float(i) for i in range(len(split_list[1:]))],split_list[1:]))
# user defined function to update the data frame with the bucket value
udf_foo = udf(lambda x: bucket_names[x], DoubleType())
bins = df_buck.withColumn("bins", udf_foo("buckets")).groupBy("bins").count().sort("bins").toPandas()


# stats

from pyspark.mllib.stat import Statistics

# select variables to check correlation
df_features = df.select("var_0","var_1","var_2","var_3") 

# create RDD table for correlation calculation
rdd_table = df_features.rdd.map(lambda row: row[0:])

# get the correlation matrix
corr_mat=Statistics.corr(rdd_table, method="pearson")

#frequency table
freq_table = df.select(col("target").cast("string")).groupBy("target").count().toPandas()

from pyspark.sql import functions as F
from pyspark.sql.functions import rank,sum,col
from pyspark.sql import Window

window = Window.rowsBetween(Window.unboundedPreceding,Window.unboundedFollowing)
# withColumn('Percent %',F.format_string("%5.0f%%\n",col('Credit_num')*100/col('total'))).\
tab = df.select(['age_class','Credit Amount']).\
   groupBy('age_class').\
   agg(F.count('Credit Amount').alias('Credit_num'),
       F.mean('Credit Amount').alias('Credit_avg'),
       F.min('Credit Amount').alias('Credit_min'),
       F.max('Credit Amount').alias('Credit_max')).\
   withColumn('total',sum(col('Credit_num')).over(window)).\
   withColumn('Percent',col('Credit_num')*100/col('total')).\
   drop(col('total'))


#correlation

from pyspark.mllib.stat import Statistics


corr_data = df.select(num_cols)

col_names = corr_data.columns
features = corr_data.rdd.map(lambda row: row[0:])
corr_mat=Statistics.corr(features, method="pearson")
corr_df = pd.DataFrame(corr_mat)
corr_df.index, corr_df.columns = col_names, col_names

print(corr_df.to_string())


# Chi-square
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest

r = ChiSquareTest.test(df, "features", "label").head()
print("pValues: " + str(r.pValues))
print("degreesOfFreedom: " + str(r.degreesOfFreedom))
print("statistics: " + str(r.statistics))

# cross table

df.stat.crosstab("age_class", "Occupation").show()

