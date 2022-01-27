package com.ysz.biz.score;

import java.time.Duration;

public class MathTmp {


  public static void main(String[] args) {
    long duration = System.currentTimeMillis() - ScoreTmp.START;

    System.out.println(ScoreTmp.log(Duration.ofHours(1000000000).toHours(), 1.5));
  }
}
