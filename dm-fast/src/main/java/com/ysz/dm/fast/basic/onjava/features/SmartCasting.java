package com.ysz.dm.fast.basic.onjava.features;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/9/27
 **/
public class SmartCasting {
  static void dumb(Object x) {
    if (x instanceof String) {
      String s = (String) x;
      if (s.length() > 0) {
        System.out.format("%d %s%n", s.length(), s.toUpperCase());
      }
    }
  }

  static void smart(Object x) {
    if (x instanceof String s && s.length() > 0) {
      System.out.format("%d %s%n", s.length(), s.toUpperCase());
    }
  }

  public static void main(String[] args) {
    dumb("dumb");
    smart("smart");
  }
}
