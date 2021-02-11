package Regression

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object GBTRegressorTest {


  def main(args: Array[String]): Unit = {


    val conf = new SparkConf().setAppName("WordCount").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sparkSession = SparkSession.builder().getOrCreate()

    // Load the data stored in LIBSVM format as a DataFrame.
    val data = sparkSession.read.format("csv")
      .option("delimiter",";")
      .option("header", "true")
      .option("inferSchema", true)
      .load("/Users/denghanbo/Documents/mldata/winequality-white.csv")
      .toDF("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "quality")
    //.toDF("features")


    data.show(false)

    //特征f1-f20合并
    val assembler = new VectorAssembler()
      .setInputCols(Array("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"))
      .setOutputCol("features")

    val data1 = assembler.transform(data)


    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data1.randomSplit(Array(0.7, 0.3))


    val dt = new GBTRegressor()
      .setLabelCol("quality")
      .setFeaturesCol("features")
      .setMaxIter(10)
      .fit(trainingData)


    val predictions = dt.transform(testData)


    predictions.show(false)


    val evaluator = new RegressionEvaluator()
      .setLabelCol("quality")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    val treeModel = dt.asInstanceOf[GBTRegressionModel]
    println("Learned regression tree model:\n" + treeModel.toDebugString)


  }


}
