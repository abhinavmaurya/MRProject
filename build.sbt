name := "mrproject"

version := "1.0"

scalaVersion := "2.11.9"
//scalaVersion := "2.12.2"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.1.0"
// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib_2.10
libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.1.0"
//libraryDependencies += "org.apache.spark" % "spark-mllib_2.12" % "2.1.0"

