package com.ysz.dm.fast.jmh;


import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

/**
 * 第一个例子、HelloWorld 没啥用
 */
public class JMHSample_01_HelloWorld {

  @Benchmark
  public String tstFormatNew() {
    return StringTools.formatWithSpecial("%s,%s,%s,abcdedfg", "1", "2", "3");
  }

  @Benchmark
  public String tstFormat() {
    return String.format("%s,%s,%s,abcdedfg", "1", "2", "3");
  }

  public static void main(String[] args) throws RunnerException {
    Options opt = new OptionsBuilder()
        .include(JMHSample_01_HelloWorld.class.getSimpleName())
        .build();

    new Runner(opt).run();
  }
}
