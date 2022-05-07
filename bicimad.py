from math import acos, cos, radians, sin
import random
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

def get_min_max_lat_lon(df):
    min_lat = df.agg({"latitude":"min"}).head()["min(latitude)"]
    max_lat = df.agg({"latitude":"max"}).head()["max(latitude)"]
    min_lon = df.agg({"longitude":"min"}).head()["min(longitude)"]
    max_lon = df.agg({"longitude":"max"}).head()["max(longitude)"]
    return min_lat, max_lat, min_lon, max_lon

def generate_points(sc, n, min_lat, max_lat, min_lon, max_lon):
    pts = []
    for _ in range(n):
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
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
        withColumnRenamed("idplug_base", "id").\
        groupBy("id").\
        count().\
        withColumnRenamed("count", "plug_count")
    unplug_count = movements_df.\
        withColumnRenamed("idunplug_base", "id").\
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

if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    stops_df = get_stops(spark, "202106")
    stops_df.show()
