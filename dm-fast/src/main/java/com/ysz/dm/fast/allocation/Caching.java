package com.ysz.dm.fast.allocation;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/9/18
 **/
public class Caching {

  private final int ARR_SIZE = 2 * 1_024 * 1_024;

  private final int[] testData = new int[ARR_SIZE];

  private void run() {
    System.err.println("Start: " + System.currentTimeMillis());

    for (int i = 0; i < 15_000; i++) {
      touchEveryItem();
      touchEveryLine();
    }

    System.err.println("Warmup finished : " + System.currentTimeMillis());
    System.err.println("Item Line");

    for (int i = 0; i < 100; i++) {
      long t0 = System.nanoTime();
      touchEveryLine();
      long t1 = System.nanoTime();
      touchEveryItem();
      long t2 = System.nanoTime();

      long elItem = t2 - t1;
      long elLine = t1 - t0;

      double diff = elItem - elLine;

      System.err.println(elItem + " " + elLine + " " + (100 * diff) / elLine);

    }
  }


  private void touchEveryItem() {
    for (int i = 0; i < testData.length; i++) {
      testData[i]++;
    }
  }


  private void touchEveryLine() {
    for (int i = 0; i < testData.length; i += 16) {
      testData[i]++;
    }
  }

  public static void main(String[] args) {
    new Caching().run();
  }

}
