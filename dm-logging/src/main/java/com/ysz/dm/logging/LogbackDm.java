package com.ysz.dm.logging;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LogbackDm {

  public static void main(String[] args) {
    Logger logger = LoggerFactory.getLogger(LogbackDm.class);
    logger.error(bigString(3_000));
//    for (int i = 0; i < 10; i++) {
//      logger.error("tst netty ......:" + i);
//    }
  }


  public static String bigString(int length) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < length; i++) {
      sb.append('x');
    }
    sb.append("" + length);

    return sb.toString();
  }
}
