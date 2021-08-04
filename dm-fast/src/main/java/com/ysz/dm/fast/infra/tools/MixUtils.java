package com.ysz.dm.fast.infra.tools;

public class MixUtils {


  public static void sleep(long millis) {
    try {
      Thread.sleep(millis);
    } catch (Exception ignored) {
    }
  }

}
