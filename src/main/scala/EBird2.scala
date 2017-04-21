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

object EBird2 {

  val usefulColumns = List.concat(List(2, 3, 5, 26, 955, 956, 957, 958, 959, 960),List(962 to 1015));
  //val usefulColumns = List(26, 955, 956, 957, 958);


  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("PageRank").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    print(usefulColumns)

    val data = sc.textFile("subset.csv")
      .map(line => helper(line))
      .filter(lp => lp!=null)

    println(data.count())
    /*data.map(x => {
      println(x.label);
      println(x.features);
    })*/

    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 5 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "variance"
    val maxDepth = 8
    val maxBins = 32

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
    model.save(sc, "model")
  }

  def helper(x: String): LabeledPoint = {

    val data = x.split(",")
    if(data(0) == "SAMPLING_EVENT_ID"){
      null;
    }else {
      val features: Array[Double] = Array.ofDim[Double](usefulColumns.length)

      var arrayIndex: Int = 0
      var featuresIndex: Int = 0

      data.foreach(r => {

        if (usefulColumns.contains(arrayIndex)) {
          features(featuresIndex) = checkValue(r, arrayIndex)
          featuresIndex += 1
        }
        arrayIndex += 1
      })

      /*println(features(0));
      features.tail.foreach(x => print(x + " "))*/
      return LabeledPoint(features(0), Vectors.dense(features.tail))
    }
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

}
