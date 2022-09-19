package com.ysz.dm.fast.perf.perfbook.c2;

import java.util.Random;

/**
 * @author carl
 * @create 2022-09-16 4:25 PM
 **/
public class MicroTest {

  private volatile double l;

  private int nLoops;

  private int[] input;

  public static void main(String[] args) {
    /*0: 参数是*/
    MicroTest ft = new MicroTest(Integer.parseInt(args[0]));
    /*1. warm*/
    ft.doTest(true);
    /*2. cold*/
    ft.doTest(false);
  }

  private MicroTest(int n) {
    this.nLoops = n;
    input = new int[nLoops];

    Random r = new Random();
    for (int i = 0; i < nLoops; i++) {
      input[i] = r.nextInt(100);
    }
  }


  private void doTest(boolean isWarmUp) {
    long then = System.currentTimeMillis();

    for (int i = 0; i < nLoops; i++) {
      this.l = fibImpl(input[i]);
    }

    if (!isWarmUp) {
      long now = System.currentTimeMillis();
      System.out.println("Elapsed time:" + (now - then));
    }
  }


  private double fibImpl(int n) {
    if (n < 0) {
      throw new IllegalArgumentException("Must be > 0");
    }

    if (n == 0) {
      return 0d;
    }

    if (n == 1) {
      return 1d;
    }

    double d = fibImpl(n - 2) + fibImpl(n - 1);
    if (Double.isInfinite(d)) {
      throw new ArithmeticException("Overload");
    }

    return d;
  }

}
