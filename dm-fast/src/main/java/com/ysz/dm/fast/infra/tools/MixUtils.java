package com.ysz.dm.fast.infra.tools;

import java.text.SimpleDateFormat;
import java.util.Date;

public class MixUtils {


  public static void sleep(long millis) {
    try {
      Thread.sleep(millis);
    } catch (Exception ignored) {
    }
  }

  public static void simpleShow(String msg) {
    System.err.printf("%s:%s\n",
        new SimpleDateFormat("HH:mm:ss.SSS").format(new Date()),
        msg
    );

  }

}
