/**
  * Created by abhinavmaurya on 4/19/17.
  */

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType}
object EBird {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("PageRank").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    // Load and parse the data file.
    //val data = sc.textFile("/Users/vikasjanardhanan/courses/mreduce/project/core-covariate_data.csv")

    //val parsedData = data.map { line =>
    //  val parts = line.split(',')
    //  LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    //}
    //print(parsedData);

    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load("/Users/vikasjanardhanan/courses/mreduce/project/core-covariate_data.csv")

    val samplingIndexer = new StringIndexer()
      .setInputCol("SAMPLING_EVENT_ID")
      .setOutputCol("SAMPLING_EVENT_ID_INDEX")
      .fit(df)

    val indexed = samplingIndexer.transform(df).toDF()

    val LocIDIndexed = new StringIndexer()
      .setInputCol("LOC_ID")
      .setOutputCol("LOC_ID_INDEX")
      .fit(indexed)

    val dfLOCID = LocIDIndexed.transform(indexed).toDF()

    val BaileyIndexed = new StringIndexer()
      .setInputCol("BAILEY_ECOREGION")
      .setOutputCol("BAILEY_ECOREGION_INDEX")
      .fit(dfLOCID)

    val dfWithBailey = BaileyIndexed.transform(dfLOCID).toDF()
    val finalDF = dfWithBailey.drop("SAMPLING_EVENT_ID").drop("LOC_ID").drop("BAILEY_ECOREGION")
    finalDF.show
    //print(df)
    // Split the data into training and test sets (30% held out for testing)
    //val splits = df.randomSplit(Array(0.7, 0.3))
     //val (trainingData, testData) = (splits(0), splits(1))

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
     //val numClasses = 2
     //val categoricalFeaturesInfo = Map[Int, Int]()
     //val numTrees = 3 // Use more in practice.
     //val featureSubsetStrategy = "auto" // Let the algorithm choose.
     //val impurity = "variance"
     //val maxDepth = 4
     //val maxBins = 32

    //val dfRDD = trainingData.rdd
     //val model = RandomForest.trainRegressor(dfRDD, categoricalFeaturesInfo,
     //  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // // Evaluate model on test instances and compute test error
    // val labelsAndPredictions = testData.map { point =>
    //   val prediction = model.predict(point.features)
    //   (point.label, prediction)
    // }
    // val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
    // println("Test Mean Squared Error = " + testMSE)
    // println("Learned regression forest model:\n" + model.toDebugString)

    // Save and load model
    //model.save(sc, "target/tmp/myRandomForestRegressionModel")
    //val sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestRegressionModel")
  }
  }
