// Databricks notebook source
// MAGIC %md
// MAGIC Meghna Jaglan
// MAGIC ### Project: Large Scale Linear Regression

// COMMAND ----------

import org.apache.spark.ml.linalg.Vectors
import scala.util.Random
import org.apache.spark.ml.linalg.Vector
import spark.implicits._
import org.apache.spark.ml.linalg.{Vector => MLVector}
import breeze.linalg.{Vector => BreezeVector, DenseVector => BreezeDenseVector, DenseMatrix => BreezeDenseMatrix}
import breeze.linalg.inv


/*
* To generate random sample data
*/
val m = 5000   // Total number of samples
val n = 20    // Total number of features

// defining range for generating random values
val rangeMin = 0  
val rangeMax = 10

// function to generate random values
def randomInRange(min: Double, max: Double): Double = min + (max - min) * Random.nextDouble()

// COMMAND ----------

// MAGIC %md
// MAGIC **Part 1** \
// MAGIC Implement closed form solution when m(number of examples is large) and n(number of features) is small:
// MAGIC \\[ \scriptsize \mathbf{\theta=(X^TX)}^{-1}\mathbf{X^Ty}\\]
// MAGIC Here, X is a distributed matrix.

// COMMAND ----------

/* 
* STEP 1: Create an example RDD for matrix X and vector y
*/

// Generate random data for X and y
val randomDataX = Seq.fill(m)(Vectors.dense(Array.fill(n)(randomInRange(rangeMin,rangeMax))))
val randomDataY = Seq.fill(m)(randomInRange(rangeMin,rangeMax))

// Convert to RDD
val rddX = sc.parallelize(randomDataX)
val rddY = sc.parallelize(randomDataY)

// COMMAND ----------

/*
* STEP 2: Compute (X^T * X) using outer product method
*/

// Function to convert from Spark Vector to Breeze Vector
def convertToBreeze(vector: MLVector): BreezeVector[Double] = BreezeDenseVector(vector.toArray)

// Function to compute outer product
def computeOuterProduct(vector: BreezeVector[Double]): BreezeDenseMatrix[Double] = {
  val matrix = BreezeDenseMatrix.zeros[Double](vector.length, vector.length)
  for (i <- 0 until vector.length) {
    for (j <- 0 until vector.length) {
      matrix(i, j) = vector(i) * vector(j)
    }
  }
  matrix
}

// Calculate X^T * X
val productXtX = rddX.map { x =>
  val bx = convertToBreeze(x)
  computeOuterProduct(bx)
}.reduce(_ + _)

/*
* STEP 3: Convert the Inverse of the Result Breeze Matrix
*/

val inverseProduct = inv(productXtX)

/*
* STEP 4: Compute (X^T * y)
*/
val productXtY = rddX.zip(rddY).map { case (x, y) => 
  val bx = convertToBreeze(x)
  bx * y 
}.reduce(_ + _)

/*
* STEP 5: Multiply (X^T X)^{-1} with X^T y
*/

val theta = inverseProduct * productXtY

// COMMAND ----------

// MAGIC %md
// MAGIC **Part 2** \
// MAGIC Implement the gradient descent update for linear regression is: \\[ \scriptsize \mathbf{\theta}_{j+1} = \mathbf{\theta}_j - \alpha \sum_i (\mathbf{\theta}_i^\top\mathbf{x}^i  - y^i) \mathbf{x}^i \\]

// COMMAND ----------

/*
* Generate Sample data
*/
val randomData = (1 to m).map { _ =>
  (Vectors.dense(Array.fill(n)(randomInRange(rangeMin,rangeMax))), randomInRange(rangeMin,rangeMax))
}
val randomDataRDD = sc.parallelize(randomData)

// COMMAND ----------

/*
* STEP 1: Initialize the elements of vector
*/
var theta = BreezeDenseVector.zeros[Double](n)
val alpha = 0.001


/*
* STEP 2: Implement a function that computes the summand
*/
def computeToBreeze(v: Vector): BreezeVector[Double] = BreezeDenseVector(v.toArray)
def calculateSummand(x: Vector, y: Double, theta: BreezeVector[Double]): BreezeVector[Double] = {
  computeToBreeze(x) * ((theta dot computeToBreeze(x)) - y)
}

// Testing the computeSummand function
val exampleXVector = Vectors.dense(Array.fill(n)(randomInRange(rangeMin,rangeMax)))
val exampleY = randomInRange(rangeMin,rangeMax)
val testSummand = calculateSummand(exampleXVector, exampleY, theta)
println(s"Testing the Summand function: $testSummand")
println()

// COMMAND ----------

/*
* STEP 3: Implement a function to compute RMSE
*/
def calculateRMSE(dataset: RDD[(Double, Double)]): Double = {
  math.sqrt(dataset.map { case (actual, predicted) => math.pow(actual - predicted, 2) }.mean())
}

// Testing the RMSE function
val sampleSize = 10
val exampleRMSEData = sc.parallelize(Seq.fill(sampleSize)((randomInRange(rangeMin,rangeMax), randomInRange(rangeMin,rangeMax))))
val testRMSE = calculateRMSE(exampleRMSEData)
println(s"Testing the RMSE function: $testRMSE")
println()

// COMMAND ----------

/*
* STEP 4: Implement a function to compute Gradient Descent
*/
def calculateGradientDescent(dataset: RDD[(Vector, Double)], theta: BreezeVector[Double], alpha: Double, iterations: Int): (BreezeVector[Double], Array[Double]) = {
  var currentTheta = theta
  val trainingErrorsPerIteration = Array.fill(iterations)(0.0)

  for (i <- 0 until iterations) {
    val totalCount = dataset.count.toDouble
    val gradientSum = dataset.map { case (x, y) => calculateSummand(x, y, currentTheta) }.reduce(_ + _).map(_ / totalCount)
    currentTheta -= alpha * gradientSum

    val predictions = dataset.map { case (x, y) => (y, (currentTheta dot computeToBreeze(x))) }
    trainingErrorsPerIteration(i) = calculateRMSE(predictions)
  }

  (currentTheta, trainingErrorsPerIteration)
}

// Running Gradient Descent for 5 iterations
val (finalTheta, trainingErrors) = calculateGradientDescent(randomDataRDD, theta, alpha, 5)

//Error Progression
trainingErrors.zipWithIndex.foreach { case (error, index) => println(s"Iteration ${index + 1}: Error (RMSE) = $error") }
println()
