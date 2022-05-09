from math import acos, cos, radians, sin
import random
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator


def get_min_max_lat_lon(df):
    min_lat = df.agg({"latitude":"min"}).head()["min(latitude)"]
    max_lat = df.agg({"latitude":"max"}).head()["max(latitude)"]
    min_lon = df.agg({"longitude":"min"}).head()["min(longitude)"]
    max_lon = df.agg({"longitude":"max"}).head()["max(longitude)"]
    return min_lat, max_lat, min_lon, max_lon

def generate_points(sc, n, mins_maxs): # mins_maxs = (min_lat, max_lat, min_lon, max_lon)
    pts = []
    for _ in range(n):
        lat = random.uniform(mins_maxs[0], mins_maxs[1])
        lon = random.uniform(mins_maxs[2], mins_maxs[3])
        pts.append((lat, lon))
    return sc.parallelize(pts)

def number_of_close_stops(df, pt):
    pass

def get_coords_from_stopid(df, stopid):
    row = df.filter(df.id == stopid).head()
    return (row["latitude"], row["longitude"])

def distance_sphere(pt1, pt2): # GREAT CIRCLE FORMULA, returns distance in km
    pt1 = tuple(map(radians, pt1))
    pt2 = tuple(map(radians, pt2))
    return 6371 * (acos(sin(pt1[0]) * sin(pt2[0]) + cos(pt1[0]) * cos(pt2[0]) * cos(pt1[1] - pt2[1])))

def get_stop_trafic(spark, filename):
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

def get_stops(spark, month_ref):
    df = spark.read.json(f"data/{month_ref}_stations.json")
    stations_df = spark.\
        createDataFrame(df.first()["stations"]).\
        select(col("id"), col("latitude").cast("float"), col("longitude").cast("float"))
    trafic_df = get_stop_trafic(spark, f"data/{month_ref}_movements.json")
    stops_df = stations_df.join(trafic_df, on=["id"], how="left_outer")
    return stops_df

def get_learning_df(df):
    #df = sc.parallelize(df).toDF()
    learning_df = df.rdd.map(lambda stop: n_closest_stops(df, stop))
    return learning_df.toDF()
    
def n_closest_stops(df, stop, n = 3):
    df2 = df.alias('df2') #copia
    distances = df2.where(df2.id != stop[0]).map(lambda stop2: (distance_sphere([stop[1], stop[2]], [stop2[1], stop2[2]])))
    out = distances.orderBy(df2[0]).take(3).flatten
    return out


def n_closest_stops_NEW(df, stopid, n=3):
    ptid = get_coords_from_stopid(df, stopid)
    return df.rdd.\
        sortBy(lambda row: distance_sphere(ptid, (row["latitude"], row["longitude"]))).\
        take(n+1)[1:]


def get_learning_df2(df):
    df = np.array(df.collect())
    print(df.shape)
    print(df[:,3]) #revisar!!
    # df = np.array([i for i in df if i[3] != None]) #dropea filas con None
    print(df.shape)
    learning_df = []
    for i in df:
        row = n_closest_stops2(df, i)
        row.append(i[3])
        learning_df.append(row)
    return learning_df
    
def n_closest_stops2(df, stop, n = 3):
    df2 = np.copy(df) #copia
    dists = lambda stop2: [distance_sphere([stop[1], stop[2]], [stop2[1], stop2[2]]), stop2[3]]
    d = np.array([dists(i) for i in df2])
    d = d[d[:,0].argsort()]
    #if stop[0] == 50:
    #    print(d)
    return d[1:n+1,:].flatten().tolist()
    
    

if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    stops_df = get_stops(spark, "202106")
    stops_df.show()
    learning_df = pd.DataFrame(get_learning_df2(stops_df))
    learning_df = spark.createDataFrame(learning_df, schema = ["d1", "t1", "d2", "t2", "d3", "t3", "t"])
    #learning_df.show(3)
    
    vectorAssembler = VectorAssembler(inputCols = ['d1', 't1', 'd2', 't2', 'd3', 't3'], outputCol = 'features')
    learning_df = vectorAssembler.transform(learning_df)
    learning_df = learning_df.select(['features', 't'])
    #learning_df.show(3)
    
    splits = learning_df.randomSplit([0.7, 0.3])
    train_df = splits[0]
    test_df = splits[1]
    
    lr = LinearRegression(featuresCol = 'features', labelCol='t', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train_df)

    lr_predictions = lr_model.transform(test_df)
    lr_predictions.select("prediction","t","features").show(5)
    lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="t",metricName="r2")
    print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
