/*
 * CSE 6332: Cloud Computing and Big Data
 * ASSIGNMENT 1: Matrix Multiplication on Hadoop
 * 
 * Author: Meghna Jaglan
 */

import java.io.*;
import java.util.*;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;

class Elem implements Writable {
    //short tag;  // 0 for M, 1 for N
    int tag;  // 0 for M, 1 for N
    int index;  // one of indexes (the other is used as a key)
    double value;

    Elem() {}
	
	Elem(int tag, int index, double value) {
		this.tag = tag;
		this.index = index;
		this.value = value;
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		tag = in.readInt();
		index = in.readInt();
		value = in.readDouble();
	}
	
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(tag);
		out.writeInt(index);
		out.writeDouble(value);
	}
}


class Pair implements WritableComparable<Pair> {
    public int i;
    public int j;
	
    Pair () {}
    Pair ( int i, int j ) { this.i = i; this.j = j; }

    @Override
	public void readFields(DataInput in) throws IOException {
		i = in.readInt();
		j = in.readInt();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(i);
		out.writeInt(j);
	}	
	
	@Override
	public int compareTo(Pair pr) {
		if (i != pr.i) {
            return (i > pr.i) ? 1 : -1;
        }
        return (j > pr.j) ? 1 : (j < pr.j) ? -1 : 0;
    }
	
	@Override
	public String toString() {
		return i + " " + j + " ";
	}
}

public class Multiply extends Configured implements Tool {

    // Matrix M Mapper
    public static class MatrixMMapper extends Mapper<Object, Text, IntWritable, Elem> {
        @Override
        public void map(Object keySource, Text valueSource, Context contextOutput) throws IOException, InterruptedException {
            String[] parts = valueSource.toString().split(",");
            int row = Integer.parseInt(parts[0]);
            Elem elem = new Elem(0, row, Double.parseDouble(parts[2]));
            contextOutput.write(new IntWritable(Integer.parseInt(parts[1])), elem);
        }
    }

    // Matrix N Mapper
    public static class MatrixNMapper extends Mapper<Object, Text, IntWritable, Elem> {
        @Override
        public void map(Object keySource, Text valueSource, Context contextOutput) throws IOException, InterruptedException {
            String[] parts = valueSource.toString().split(",");
            int col = Integer.parseInt(parts[1]);
            Elem elem = new Elem(1, col, Double.parseDouble(parts[2]));
            contextOutput.write(new IntWritable(Integer.parseInt(parts[0])), elem);
        }
    }

    // Intermediate Reducer
    public static class IntermediateReducer extends Reducer<IntWritable, Elem, Pair, DoubleWritable> {
        @Override
        public void reduce(IntWritable keySource, Iterable<Elem> valueSource, Context contextOutput) throws IOException, InterruptedException {
            List<Elem> mList = new ArrayList<>();
            List<Elem> nList = new ArrayList<>();
            
            for (Elem v : valueSource) {
                if (v.tag == 0) mList.add(new Elem(v.tag, v.index, v.value));
                else nList.add(new Elem(v.tag, v.index, v.value));
            }
            
            for (Elem m : mList) {
                for (Elem n : nList) {
                    contextOutput.write(new Pair(m.index, n.index), new DoubleWritable(m.value * n.value));
                }
            }
        }
    }
	
    //Second Mapper for Matrix Multiplication
	public static class FinalMNMatrixMapper extends Mapper<Object, Text, Pair, DoubleWritable> {
		@Override
		public void map(Object keySource, Text valueSource, Context contextOutput) throws IOException, InterruptedException {
            String row = valueSource.toString();
            String[] matrixEntries = row.split("\\s+");
            Pair matrixPair = new Pair(Integer.valueOf(matrixEntries[0]), Integer.valueOf(matrixEntries[1]));
            DoubleWritable cellValue = new DoubleWritable(Double.valueOf(matrixEntries[2]));
            contextOutput.write(matrixPair, cellValue);
        }
	}
	
    // Final Reducer
    public static class FinalReducer extends Reducer<Pair, DoubleWritable, Pair, DoubleWritable> {
        @Override
        public void reduce(Pair key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
            double sum = 0;
            for (DoubleWritable val : values) {
                sum += val.get();
            }
            context.write(key, new DoubleWritable(sum));
        }
    }

    @Override
    public int run ( String[] args ) throws Exception {
        /* ... */
        return 0;
    }

    public static void main(String[] args) throws Exception {
        // First Map-Reduce Job (Intermediate calculations)
        Job intermediateJob = Job.getInstance();
        intermediateJob.setJobName("MatrixIntermediateOperations");
        intermediateJob.setJarByClass(Multiply.class);

        MultipleInputs.addInputPath(intermediateJob, new Path(args[0]), TextInputFormat.class, MatrixMMapper.class);
        MultipleInputs.addInputPath(intermediateJob, new Path(args[1]), TextInputFormat.class, MatrixNMapper.class);
        intermediateJob.setReducerClass(IntermediateReducer.class);

        intermediateJob.setMapOutputKeyClass(IntWritable.class);
        intermediateJob.setMapOutputValueClass(Elem.class);

        intermediateJob.setOutputKeyClass(Pair.class);
        intermediateJob.setOutputValueClass(DoubleWritable.class);

        // Directing intermediate results to a specified path
        intermediateJob.setOutputFormatClass(TextOutputFormat.class);
        FileOutputFormat.setOutputPath(intermediateJob, new Path(args[2]));

        intermediateJob.waitForCompletion(true);

        // Second Map-Reduce Job (Final results)
        Job finalJob = Job.getInstance();
        finalJob.setJobName("MatrixFinalOperations");
        finalJob.setJarByClass(Multiply.class);

        finalJob.setMapperClass(FinalMNMatrixMapper.class);
        finalJob.setReducerClass(FinalReducer.class);

        finalJob.setMapOutputKeyClass(Pair.class);
        finalJob.setMapOutputValueClass(DoubleWritable.class);

        finalJob.setOutputKeyClass(Pair.class);
        finalJob.setOutputValueClass(DoubleWritable.class);

        // Setting up input (intermediate results) and output paths
        finalJob.setInputFormatClass(TextInputFormat.class);
        finalJob.setOutputFormatClass(TextOutputFormat.class);
        
        FileInputFormat.setInputPaths(finalJob, new Path(args[2]));
        FileOutputFormat.setOutputPath(finalJob, new Path(args[3]));

        finalJob.waitForCompletion(true);
    }


}
