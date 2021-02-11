package classer

import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object Other {


  def main(args: Array[String]): Unit = {


    val conf = new SparkConf().setAppName("WordCount").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sparkSession =  SparkSession.builder().getOrCreate()

    // Load the data stored in LIBSVM format as a DataFrame.
    val data = sparkSession.read.format("csv")
        .option("delimiter", "\t")
        .option("inferSchema",true)
        .load("/Users/denghanbo/Documents/mldata/inputdata3.txt")
        .toDF("lable","f1","f2","f3")


    data.show()


  }



}
