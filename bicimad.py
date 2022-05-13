from math import acos, cos, radians, sin
import random
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

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
        row = n_closest_stops(df, (i[1], i[2]))
        row.append(i[3])
        row.insert(0, i[2])
        row.insert(0, i[1])
        learning.append(row)
    return spark.createDataFrame(pd.DataFrame(learning), schema=["lat", "lon", "dist*trafic1", "dist*trafic2", "dist*trafic3", "trafic"])
    
def n_closest_stops(df, pt, n=3):
    df2 = np.copy(df)
    dists = lambda stop: [distance_sphere(pt, [stop[1], stop[2]]) * stop[3]]
    d = np.array([dists(i) for i in df2])
    d = d[d[:,0].argsort()]
    return d[1:n+1,:].flatten().tolist()
    
def get_min_max_lat_lon(df):
    min_lat = df.agg({"latitude":"min"}).head()["min(latitude)"]
    max_lat = df.agg({"latitude":"max"}).head()["max(latitude)"]
    min_lon = df.agg({"longitude":"min"}).head()["min(longitude)"]
    max_lon = df.agg({"longitude":"max"}).head()["max(longitude)"]
    return min_lat, max_lat, min_lon, max_lon

def pt_in_hull(pt, hull, eps=10**(-4)):
    return all((np.dot(eq[:-1], pt) + eq[-1] <= eps) for eq in hull.equations)

def generate_points(sc: SparkContext, n, df): # mins_maxs = (min_lat, max_lat, min_lon, max_lon)
    pts = []
    i = 0
    hull = ConvexHull(np.array(df.select(col("latitude"), col("longitude")).collect()))
    min_lat, max_lat, min_lon, max_lon = get_min_max_lat_lon(df)
    while i < n:
        pt = (random.uniform(min_lat, max_lat), random.uniform(min_lon, max_lon))
        if pt_in_hull(pt, hull) and pt not in pts:
            pts.append(pt)
            i += 1
    return sc.parallelize(pts)

def get_mesh(sc: SparkContext, n, df):
    pts = np.array(generate_points(sc, n, df).collect())
    df_ = np.array(df.where(col("trafic").isNotNull()).collect())
    testing = []
    for pt in pts:
        row = n_closest_stops(df_, pt)
        row.insert(0, pt[1])
        row.insert(0, pt[0])
        testing.append(row)
    return spark.createDataFrame(pd.DataFrame(testing), schema=["lat", "lon", "dist*trafic1", "dist*trafic2", "dist*trafic3"])

    
if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    stops_df = get_stops(spark, "202106")
    stops_df.show()

    # TRAINING THE MODEL -------------------------------------------------------------------------

    learning_df = get_learning_df(stops_df)
    feature_assembler = VectorAssembler(inputCols=["lat", "lon", "dist*trafic1", "dist*trafic2", "dist*trafic3"], outputCol="features")
    output = feature_assembler.transform(learning_df)

    final_data = output.select(["features", "lat", "lon", "trafic"])
    train_data, test_data = final_data.randomSplit([0.7, 0.3])

    lr = LinearRegression(featuresCol="features", labelCol="trafic", maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train_data)

    lr_predictions = lr_model.transform(test_data)
    lr_predictions.show()
    lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="trafic",metricName="r2")
    print(f"R Squared (R2) on test data = {lr_evaluator.evaluate(lr_predictions)}")

    # TESTING THE MODEL --------------------------------------------------------------------------

    mesh = get_mesh(sc, 10**5, stops_df)
    mesh_test = feature_assembler.transform(mesh).select(["features", "lat", "lon"])
    mesh_predictions = lr_model.transform(mesh_test)
    mesh_predictions.show()

    max_trafic_pred = mesh_predictions.agg({"prediction": "max"}).head()[0]
    best_row = mesh_predictions.filter(col("prediction") == max_trafic_pred).head()
    print(f"Un punto candidato a poner una nueva parada de Bicimad es ({best_row['lat']}, {best_row['lon']}) puesto que se estima que tenga un total de {best_row['prediction']} usos al mes")
