from pyspark.sql import SparkSession

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

from JobData import JobData

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("ALSExample")\
        .getOrCreate()

    lines = spark.read.option("header", "true").csv("jobData/rating.csv").rdd

    ratingsRDD = lines.map(lambda p: Row(userId=int(p[0]), jobId=int(p[1]),
                                         rating=float(p[2])))
    
    ratings = spark.createDataFrame(ratingsRDD)
    
    (training, test) = ratings.randomSplit([0.8, 0.2])

    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="jobId", ratingCol="rating",
              coldStartStrategy="drop")
    model = als.fit(training)

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

    userRecs = model.recommendForAllUsers(10)
    
    user85Recs = userRecs.filter(userRecs['userId'] == 1).collect()
    
    spark.stop()

    jb = JobData()
    jb.LoadJobData()
        
    for row in user85Recs:
        for rec in row.recommendations:
            print(jb.getJobName(rec.movieId))

