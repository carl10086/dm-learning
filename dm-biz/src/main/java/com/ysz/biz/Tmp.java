package com.ysz.biz;

import org.apache.commons.lang3.time.FastDateFormat;

public class Tmp {


  public static void main(String[] args) throws Exception {
    final long now = FastDateFormat.getInstance("yyyy-MM-dd HH:mm:ss").parse("2022-01-24 14:56:30").getTime();
    final long createAt = FastDateFormat.getInstance("yyyy-MM-dd HH:mm:ss").parse("2022-01-24 13:39:30").getTime();

    long dur = now - createAt;

    System.out.println(33L * now / dur);
  }

}
