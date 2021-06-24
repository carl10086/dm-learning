package com.ysz.dm.web.common;

public class Utils {

  public static void trySleep(long mills) {
    try {
      Thread.sleep(mills);
    } catch (InterruptedException ignored) {
    }
  }

}
