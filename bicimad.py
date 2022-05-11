from math import acos, cos, radians, sin
import random
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator


# def get_min_max_lat_lon(df):
#     min_lat = df.agg({"latitude":"min"}).head()["min(latitude)"]
#     max_lat = df.agg({"latitude":"max"}).head()["max(latitude)"]
#     min_lon = df.agg({"longitude":"min"}).head()["min(longitude)"]
#     max_lon = df.agg({"longitude":"max"}).head()["max(longitude)"]
#     return min_lat, max_lat, min_lon, max_lon

# def generate_points(sc, n, mins_maxs): # mins_maxs = (min_lat, max_lat, min_lon, max_lon)
#     pts = []
#     for _ in range(n):
#         lat = random.uniform(mins_maxs[0], mins_maxs[1])
#         lon = random.uniform(mins_maxs[2], mins_maxs[3])
#         pts.append((lat, lon))
#     return sc.parallelize(pts)

# def get_coords_from_stopid(df, stopid):
#     row = df.filter(df.id == stopid).head()
#     return (row["latitude"], row["longitude"])

def distance_sphere(pt1, pt2): # GREAT CIRCLE FORMULA, returns distance in km
    pt1 = tuple(map(radians, pt1))
    pt2 = tuple(map(radians, pt2))
    x = sin(pt1[0]) * sin(pt2[0]) + cos(pt1[0]) * cos(pt2[0]) * cos(pt1[1] - pt2[1])
    if x > 1: # redondeamos el error para no salirnos del domino del arccos
        x = 1
    return 6371 * (acos(x))

def get_stop_trafic(spark: SparkSession, filename):
    movements_df = spark.read.json(filename)
    plug_count = movements_df.\
        withColumnRenamed("idplug_station", "id").\
        groupBy("id").\
        count().\
        withColumnRenamed("count", "plug_count")
    unplug_count = movements_df.\
        withColumnRenamed("idunplug_station", "id").\
        groupBy("id").\
        count().\
        withColumnRenamed("count", "unplug_count")
    trafic_df = plug_count.\
        join(unplug_count, on=["id"], how="left_outer").\
        sort("id").\
        withColumn("trafic", col("plug_count")+col("unplug_count")).\
        select(col("id"), col("trafic"))
    return trafic_df

def get_stops(spark: SparkSession, month_ref):
    df = spark.read.json(f"data/{month_ref}_stations.json")
    stations_df = spark.\
        createDataFrame(df.first()["stations"]).\
        select(col("id"), col("latitude").cast("float"), col("longitude").cast("float"))
    trafic_df = get_stop_trafic(spark, f"data/{month_ref}_movements.json")
    stops_df = stations_df.join(trafic_df, on=["id"], how="left_outer")
    return stops_df

def get_learning_df(df):
    df = np.array(df.where(col("trafic").isNotNull()).collect())
    learning = []
    for i in df:
        row = n_closest_stops(df, i)
        row.append(i[3])
        learning.append(row)
    return spark.createDataFrame(pd.DataFrame(learning), schema=["dist1", "trafic1", "dist2", "trafic2", "dist3", "trafic3", "trafic"])
    
def n_closest_stops(df, stop, n = 3):
    df2 = np.copy(df)
    dists = lambda stop2: [distance_sphere([stop[1], stop[2]], [stop2[1], stop2[2]]), stop2[3]]
    d = np.array([dists(i) for i in df2])
    d = d[d[:,0].argsort()]
    return d[1:n+1,:].flatten().tolist()
    
    
if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    stops_df = get_stops(spark, "202106")
    stops_df.show()

    learning_df = get_learning_df(stops_df)
    feature_assembler = VectorAssembler(inputCols=["dist1", "trafic1", "dist2", "trafic2", "dist3", "trafic3"], outputCol="features")
    output = feature_assembler.transform(learning_df)

    final_data = output.select(["features", "trafic"])
    train_data, test_data = final_data.randomSplit([0.7, 0.3])

    lr = LinearRegression(featuresCol="features", labelCol="trafic", maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train_data)

    lr_predictions = lr_model.transform(test_data)
    lr_predictions.show()
    lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="trafic",metricName="r2")
    print(f"R Squared (R2) on test data = {lr_evaluator.evaluate(lr_predictions)}")
