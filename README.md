This repository contains the projects completed as part of the Cloud Computing and Big Data course (CSE 6332) course, part of my Master of Science in Computer Science program. The course covers various aspects of cloud computing and big data technologies, including distributed computing, data processing frameworks, and cloud platforms.

## Projects

### Project 1: Matrix Multiplication using MapReduce

#### Description
This project involves implementing matrix multiplication using the MapReduce programming model. The goal is to understand the basics of distributed computing and the challenges associated with processing large-scale data using cloud computing technologies.

#### Files
- `CSE6332_Project1_Multiply.java`

#### Implementation Details
- **Objective**: Implement a matrix multiplication algorithm using the MapReduce framework in Java.
- **Technologies Used**: Java, Hadoop MapReduce.
- **Key Concepts**: 
  - Basics of the MapReduce programming model.
  - Distributed computing principles.
  - Handling large-scale data processing in a cloud environment.
  
#### Steps
1. **Matrix Initialization**: Initialize matrices with random values or predefined values for testing.
2. **Map Phase**: Implement the map function to process input matrix elements and produce intermediate key-value pairs.
3. **Shuffle and Sort Phase**: Use Hadoop's built-in mechanisms to shuffle and sort the intermediate data.
4. **Reduce Phase**: Implement the reduce function to aggregate the intermediate data and produce the final product matrix.
5. **Result Compilation**: Combine the results from the reduce phase to form the final product matrix.


### Project 2: Big Data Processing with Scala and Spark on Databricks

#### Description
This project involves processing large datasets using Scala and Apache Spark on the Databricks platform. The aim is to gain hands-on experience with big data frameworks and learn how to efficiently process and analyze big data.

#### Files
- `CSE6332_Project2.scala`

#### Implementation Details
- **Objective**: Implement data processing and analysis tasks using Scala and Spark on Databricks.
- **Technologies Used**: Scala, Apache Spark, Databricks.
- **Key Concepts**:
  - DataFrame and RDD operations in Spark.
  - Functional programming with Scala.
  - Data processing and transformation on a cloud-based platform.
  
#### Steps
1. **Data Loading**: Load large datasets into Spark DataFrames or RDDs on Databricks.
2. **Data Transformation**: Apply various transformations like filtering, mapping, and aggregations to process the data.
3. **Analysis**: Perform data analysis tasks such as calculating statistics, grouping data, and summarizing results.
4. **Optimization**: Optimize the Spark jobs to improve performance and reduce execution time.
5. **Results**: Output the results to the console or save them to external storage like Databricks DBFS (Databricks File System) or a database.
