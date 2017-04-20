/**
  * Created by abhinavmaurya on 4/19/17.
  */

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType}
object EBird2 {

  def main(args: Array[String]) {

    var conf = new SparkConf()
      .setAppName("EBird");
    var sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    // Load and parse the data file.
    //val data = sc.textFile("/Users/vikasjanardhanan/courses/mreduce/project/core-covariate_data.csv")

    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load("/Users/vikasjanardhanan/courses/mreduce/project/core-covariate_data.csv")

    print(df)
    // Split the data into training and test sets (30% held out for testing)
    //val splits = data.randomSplit(Array(0.7, 0.3))
    // val (trainingData, testData) = (splits(0), splits(1))

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    //val numClasses = 2
    // val categoricalFeaturesInfo = Map[Int, Int]()
    // val numTrees = 3 // Use more in practice.
    // val featureSubsetStrategy = "auto" // Let the algorithm choose.
    // val impurity = "variance"
    // val maxDepth = 4
    // val maxBins = 32

    // val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
    //   numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

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
