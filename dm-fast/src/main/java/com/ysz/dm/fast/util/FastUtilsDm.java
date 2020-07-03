package com.ysz.dm.fast.util;

import org.apache.commons.lang3.time.FastDateFormat;
import org.junit.Test;

public class FastUtilsDm {

  @Test
  public void tstBigIntSet() throws Exception {
    System.out.println(FastDateFormat.getInstance("yyyyMMdd HH").parse("20200630 12").getTime());
  }

}
