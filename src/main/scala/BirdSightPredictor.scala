/**
  * Created by abhinavmaurya on 4/19/17.
  */

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType}
import org.apache.spark.sql.functions._
import com.google.common.collect.ImmutableMap;
import org.apache.spark.sql.functions.countDistinct;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

object BirdSightPredictor {


  def main(args: Array[String]) {

    /*val featuresInput = Array("LATITUDE", "LONGITUDE", "MONTH", "DAY",
      "TIME", "POP00_SQMI", "HOUSING_DENSITY", "HOUSING_PERCENT_VACANT",
      "CAUS_TEMP_AVG", "CAUS_TEMP_MIN", "CAUS_TEMP_MAX", "CAUS_PREC", "CAUS_SNOW")*/

    val featuresInput = DatasetColumns.getColumnNameList.toBuffer - (DatasetColumns.getLabelColumnName, "SAMPLING_EVENT_ID")

    /*val featuresInput = Array("LATITUDE","LONGITUDE","MONTH","DAY","TIME","POP00_SQMI","HOUSING_DENSITY","HOUSING_PERCENT_VACANT",
      "ELEV_GT","BCR","OMERNIK_L3_ECOREGION","CAUS_TEMP_AVG","CAUS_TEMP_MIN",
      "CAUS_TEMP_MAX", "CAUS_PREC","CAUS_SNOW"
      ,"NLCD2001_FS_C11_7500_PLAND",
      "NLCD2001_FS_C21_7500_PLAND","NLCD2001_FS_C22_7500_PLAND","NLCD2001_FS_C23_7500_PLAND",
      "NLCD2001_FS_C24_7500_PLAND","NLCD2001_FS_C31_7500_PLAND","NLCD2001_FS_C41_7500_PLAND",
      "NLCD2001_FS_C42_7500_PLAND","NLCD2001_FS_C43_7500_PLAND","NLCD2001_FS_C52_7500_PLAND",
      "NLCD2001_FS_C71_7500_PLAND","NLCD2001_FS_C81_7500_PLAND","NLCD2001_FS_C82_7500_PLAND",
      "NLCD2001_FS_C90_7500_PLAND","NLCD2001_FS_C95_7500_PLAND","NLCD2006_FS_C11_7500_PLAND",
      "NLCD2006_FS_C21_7500_PLAND","NLCD2006_FS_C22_7500_PLAND",
      "NLCD2006_FS_C23_7500_PLAND","NLCD2006_FS_C24_7500_PLAND","NLCD2006_FS_C31_7500_PLAND",
      "NLCD2006_FS_C41_7500_PLAND","NLCD2006_FS_C42_7500_PLAND","NLCD2006_FS_C43_7500_PLAND",
      "NLCD2006_FS_C52_7500_PLAND","NLCD2006_FS_C71_7500_PLAND","NLCD2006_FS_C81_7500_PLAND",
      "NLCD2006_FS_C82_7500_PLAND","NLCD2006_FS_C90_7500_PLAND","NLCD2006_FS_C95_7500_PLAND",
      "NLCD2011_FS_C11_7500_PLAND","NLCD2011_FS_C21_7500_PLAND",
      "NLCD2011_FS_C22_7500_PLAND","NLCD2011_FS_C23_7500_PLAND","NLCD2011_FS_C24_7500_PLAND",
      "NLCD2011_FS_C31_7500_PLAND","NLCD2011_FS_C41_7500_PLAND","NLCD2011_FS_C42_7500_PLAND",
      "NLCD2011_FS_C43_7500_PLAND","NLCD2011_FS_C52_7500_PLAND","NLCD2011_FS_C71_7500_PLAND",
      "NLCD2011_FS_C81_7500_PLAND","NLCD2011_FS_C82_7500_PLAND","NLCD2011_FS_C90_7500_PLAND",
      "NLCD2011_FS_C95_7500_PLAND")*/
    val conf = new SparkConf().setAppName("EBird")
      .setMaster("local[*]")
      //.setMaster("yarn")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    // Load and parse the data file.
    var df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("NullValue","?")
      .option("inferSchema", "true") // Automatically infer data types
      .load("subset.csv")
      //.load(args(0))

    /*
    val colNames = Array("LATITUDE", "LONGITUDE", "MONTH", "DAY",
      "TIME","Agelaius_phoeniceus", "POP00_SQMI", "HOUSING_DENSITY", "HOUSING_PERCENT_VACANT",
      "CAUS_TEMP_AVG", "CAUS_TEMP_MIN", "CAUS_TEMP_MAX", "CAUS_PREC", "CAUS_SNOW")*/


    /*
    val colNames = Array("LATITUDE","LONGITUDE","MONTH","DAY","TIME","Agelaius_phoeniceus",
      "POP00_SQMI","HOUSING_DENSITY","HOUSING_PERCENT_VACANT","ELEV_GT"
      ,"BCR","OMERNIK_L3_ECOREGION","CAUS_TEMP_AVG","CAUS_TEMP_MIN","CAUS_TEMP_MAX",
      "CAUS_PREC","CAUS_SNOW"
      ,"NLCD2001_FS_C11_7500_PLAND",
      "NLCD2001_FS_C21_7500_PLAND","NLCD2001_FS_C22_7500_PLAND","NLCD2001_FS_C23_7500_PLAND",
      "NLCD2001_FS_C24_7500_PLAND","NLCD2001_FS_C31_7500_PLAND","NLCD2001_FS_C41_7500_PLAND",
      "NLCD2001_FS_C42_7500_PLAND","NLCD2001_FS_C43_7500_PLAND","NLCD2001_FS_C52_7500_PLAND",
      "NLCD2001_FS_C71_7500_PLAND","NLCD2001_FS_C81_7500_PLAND","NLCD2001_FS_C82_7500_PLAND",
      "NLCD2001_FS_C90_7500_PLAND","NLCD2001_FS_C95_7500_PLAND","NLCD2006_FS_C11_7500_PLAND",
      "NLCD2006_FS_C21_7500_PLAND","NLCD2006_FS_C22_7500_PLAND",
      "NLCD2006_FS_C23_7500_PLAND","NLCD2006_FS_C24_7500_PLAND","NLCD2006_FS_C31_7500_PLAND",
      "NLCD2006_FS_C41_7500_PLAND","NLCD2006_FS_C42_7500_PLAND","NLCD2006_FS_C43_7500_PLAND",
      "NLCD2006_FS_C52_7500_PLAND","NLCD2006_FS_C71_7500_PLAND","NLCD2006_FS_C81_7500_PLAND",
      "NLCD2006_FS_C82_7500_PLAND","NLCD2006_FS_C90_7500_PLAND","NLCD2006_FS_C95_7500_PLAND",
      "NLCD2011_FS_C11_7500_PLAND","NLCD2011_FS_C21_7500_PLAND",
      "NLCD2011_FS_C22_7500_PLAND","NLCD2011_FS_C23_7500_PLAND","NLCD2011_FS_C24_7500_PLAND",
      "NLCD2011_FS_C31_7500_PLAND","NLCD2011_FS_C41_7500_PLAND","NLCD2011_FS_C42_7500_PLAND",
      "NLCD2011_FS_C43_7500_PLAND","NLCD2011_FS_C52_7500_PLAND","NLCD2011_FS_C71_7500_PLAND",
      "NLCD2011_FS_C81_7500_PLAND","NLCD2011_FS_C82_7500_PLAND","NLCD2011_FS_C90_7500_PLAND",
      "NLCD2011_FS_C95_7500_PLAND")*/
    //df = df.select(colNames.head,colNames.tail: _*)
    //val colNos = Array(2,3,5,6,7,26,955,956,957,958,960,962,963,964,965,966,967,968,969)++ (968 to 984).toArray ++ (986 to 1000).toArray  ++ (1002 to 1015).toArray

    //val colNos = List.concat(List(2, 3, 5,6,7,26,955,956,957,958,960),(962 to 967),(968 to 984),(986 to 1000),(1002 to 1015)).toArray;
    val colNos = DatasetColumns.getColumnIndexList.toBuffer - 0
    df = df.select(colNos map df.columns map col: _*)
    df.show
    val colsToBeImputed = Array("POP00_SQMI","HOUSING_DENSITY", "HOUSING_PERCENT_VACANT", "CAUS_TEMP_AVG", "CAUS_TEMP_MIN", "CAUS_TEMP_MAX", "CAUS_PREC","CAUS_SNOW","OMERNIK_L3_ECOREGION","BCR","ELEV_GT")

    val categoricalCol = Array("ELEV_GT","CAUS_TEMP_AVG", "CAUS_TEMP_MIN", "CAUS_TEMP_MAX", "CAUS_PREC","CAUS_SNOW")
    val imputedVal = df.select(colsToBeImputed.map(avg(_)):_*).toDF(colsToBeImputed:_*)

    val imputerMap = imputedVal.columns.zip(imputedVal.first().toSeq).
      map(a =>
        if (categoricalCol contains a._1)
          (a._1 -> a._2.toString.toDouble.ceil.toString)
        else
          (a._1 -> a._2)).toMap

    val imputedDF = df.na.fill(imputerMap)
    //imputedDF.show
    val finalDFFill = imputedDF
      .na.replace("Agelaius_phoeniceus",ImmutableMap.of("X", "1"))

    finalDFFill.printSchema()
    finalDFFill.show
   val testDF = finalDFFill.columns.foldLeft(finalDFFill)((current, c) => current.withColumn(c, col(c).cast(DoubleType)))

    // create new dataframe with added column named "notempty"
    val finalDf = testDF.withColumn("Agelaius_phoeniceus",
      when($"Agelaius_phoeniceus" > 1.0, 1.0).otherwise(0.0))
    //finalDf.show(100)
    //finalDf.printSchema
    val assembler = new VectorAssembler()
      .setInputCols(featuresInput.toArray)
      .setOutputCol("features")

    //val df2 = assembler.transform(finalDf)


    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(13)
      //.fit(df2)



    val Array(trainingData, testData) = finalDf.withColumnRenamed("Agelaius_phoeniceus","label").randomSplit(Array(0.7, 0.3))

    //featuresInput.foreach( c => {
    //  finalDf.agg(countDistinct(c)).toDF().show
    //})
    //finalDf.agg(countDistinct("Agelaius_phoeniceus")).toDF().show
     val rf = new RandomForestClassifier()
       //.setLabelCol("Agelaius_phoeniceus")
        .setLabelCol("label")
       .setFeaturesCol("indexedFeatures")
       .setNumTrees(125)
       .setMaxDepth(10)
      //.setMaxBins(125)


     val pipeline = new Pipeline()
       .setStages(Array(assembler,featureIndexer,rf))
      //.setStages(Array(assembler,rf))

     val model = pipeline.fit(trainingData)

     val predictions = model.transform(testData)



     val evaluator = new MulticlassClassificationEvaluator()
       //.setLabelCol("Agelaius_phoeniceus")
         .setLabelCol("label")
       .setPredictionCol("prediction")
       .setMetricName("accuracy")
     val accuracy = evaluator.evaluate(predictions)
     println("Test Error = " + (1.0 - accuracy))


    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.numTrees,Array(100,125))
      //.addGrid(rf.numTrees,Array(125))
      //.addGrid(rf.maxDepth, Array(10))
      .addGrid(rf.maxDepth, Array(8,10))
      .addGrid(rf.impurity, Array("entropy", "gini"))
      .build()
    /*
     val cv = new CrossValidator()
       .setEstimator(pipeline)
       .setEvaluator(new MulticlassClassificationEvaluator)
       .setEstimatorParamMaps(paramGrid)
       .setNumFolds(2)  // Use 3+ in practice
    */

/*    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      // 80% of the data will be used for training and the remaining 20% for validation.
      .setTrainRatio(0.8)

    trainingData.show
     val cvModel = trainValidationSplit.fit(trainingData)
     //val cvModel = cv.fit(trainingData)
     val predictions_cv = cvModel.transform(testData)
     val accuracy_cv = evaluator.evaluate(predictions_cv)
     println("New Accuracy after selecting best model= " + accuracy_cv)*/


    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]

    rfModel.write.overwrite().save("model_2")
    //rfModel.write.overwrite.save("model")
    //rfModel.toDebugString





    //val categoricalFeaturesInfo = Map[Int,Int] (
    //  (0,13),
    //  (9,38),
    //  (11,121)
    //)
    //val numTrees = 10 // Use more in practice.
    //val featureSubsetStrategy = "auto" // Let the algorithm choose.
    //val impurity = "gini"
    //val maxDepth = 9
    //val maxBins = 4000
  }
}
