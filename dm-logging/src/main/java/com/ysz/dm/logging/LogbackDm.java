package com.ysz.dm.logging;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LogbackDm {

  @Test
  public void tst() {
    Logger logger = LoggerFactory.getLogger(LogbackDm.class);
    logger.info("tst ......");
  }
}
