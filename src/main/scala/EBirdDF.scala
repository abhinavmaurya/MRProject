/**
  * Created by abhinavmaurya on 4/19/17.
  */

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer

object EBirdDF {

  //val usefulColumns = List.concat(List(2, 3, 5, 26, 955, 956, 957, 958, 959, 960),(962 to 1015));
  val usefulColumnsIndex = List(0, 2, 3, 5, 6, 7, 26, 955, 956, 957, 958, 960, 962, 963, 964, 965, 966, 967);
  val columnNames = Array("LABEL", "SAMPLING_EVENT_ID", "LATITUDE", "LONGITUDE", "MONTH", "DAY", "TIME",
    "POP00_SQMI", "HOUSING_DENSITY", "HOUSING_PERCENT_VACANT", "ELEV_GT", "BCR", "OMERNIK_L3_ECOREGION",
    "CAUS_TEMP_AVG","CAUS_TEMP_MIN", "CAUS_TEMP_MAX", "CAUS_PREC","CAUS_SNOW");


  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("EBird")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    print(usefulColumnsIndex.length)

    val data = sc.textFile("subset.csv")
      .mapPartitionsWithIndex({ (idx, iter) => if (idx == 0) iter.drop(1) else iter })
      .map(line => {
        val data = line.split(",")
        (
          transformLabel(data(26), 26),
          data(0),

          if (data(2).charAt(0) == '?') null else data(2),
          if (data(3).charAt(0) == '?') null else data(3),
          if (data(5).charAt(0) == '?') null else data(5),
          if (data(6).charAt(0) == '?') null else data(6),
          if (data(7).charAt(0) == '?') null else data(7),
          if (data(955).charAt(0) == '?') null else data(955),
          if (data(956).charAt(0) == '?') null else data(956),
          if (data(957).charAt(0) == '?') null else data(957),
          if (data(958).charAt(0) == '?') null else data(958),
          if (data(960).charAt(0) == '?') null else data(960),
          if (data(962).charAt(0) == '?') null else data(962),
          if (data(963).charAt(0) == '?') null else data(963),
          if (data(964).charAt(0) == '?') null else data(964),
          if (data(965).charAt(0) == '?') null else data(965),
          if (data(966).charAt(0) == '?') null else data(966),
          if (data(967).charAt(0) == '?') 0.0 else data(967)

          /*checkValue(data(2), 2),
          checkValue(data(3), 3),
          checkValue(data(5), 5),
          checkValue(data(6), 6),
          checkValue(data(7), 7),
          checkValue(data(955), 955),
          checkValue(data(956), 956),
          checkValue(data(957), 957),
          checkValue(data(958), 958),

          checkValue(data(960), 960),
          checkValue(data(962), 962),
          checkValue(data(963), 963),
          checkValue(data(964), 964),
          checkValue(data(965), 965),
          checkValue(data(966), 966),
          checkValue(data(967), 967)*/
        )
        //usefulColumnsIndex.foreach(i => checkValue(data(i), i))

      })

    //val df = data.toDF(featuresList: _*)
    println(data.count())
    val df = data.toDF(columnNames: _*)

    /*val dfId = df.withColumn("uniqueId", monotonically_increasing_id())
    dfId.show()*/

    df.show()
    df.printSchema()

    /*val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 10 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "variance"
    val maxDepth = 9
    val maxBins = 4000

    val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
    println("Test Mean Squared Error = " + testMSE)
    println("Learned regression forest model:\n" + model.toDebugString)

    // Save and load model
    model.save(sc, "model")*/
  }

  def helper(x: String): Unit = {

    val data = x.split(",")
    (
        checkValue(data(26), 26),
        checkValue(data(2), 2),
        checkValue(data(3), 3),
        checkValue(data(5), 5),
        checkValue(data(955), 955),
        checkValue(data(956), 956),
        checkValue(data(957), 957),
        checkValue(data(958), 958),
        checkValue(data(959), 959),
        checkValue(data(960), 960)
    )
  }

  def checkValue(p: String, index:Int): Double = {
    var r: Double = 0.0

    // check if column is label column
    if(index == 26){
      if(p == "X") {
        r = 1.0;
      }else if(p.toInt > 0){
        r = 1.0;
      }else {
        r = 0.0
      }
    }else{
      if(p == "?"){
        r = 0.0
      }else{
        r = p.toDouble;
      }
    }
    return r;
  }


  def transformLabel(p: String, index:Int): String = {
    var r: String = "0.0"
    // check if column is label column
      if (p == "X") {
        r = "1.0";
      } else if (p.toInt > 0) {
        r = "1.0";
      } else {
        r = "0.0"
      }

    return r;
  }

}
