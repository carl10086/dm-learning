package com.ysz.dm.fast;

/**
 * @author carl.yu
 * @date 2020/3/17
 */
public class IdMain {

  private static final String DT_APP_ENV = "DT_APP_ENV";

  private static void tstTimed() {
  }

  public static void main(String[] args) {
    System.out.println(System.getProperty("os.detected.name"));
    System.out.println(System.getProperty("os.arch"));
  }


}
