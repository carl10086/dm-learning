package com.ysz.dm.netty.timer;

import io.netty.util.HashedWheelTimer;
import io.netty.util.Timeout;
import io.netty.util.Timer;
import java.util.Date;
import java.util.concurrent.TimeUnit;
import org.apache.commons.lang.time.FastDateFormat;

public class HashedWheelTimer_Dm_001 {

  public static void main(String[] args) throws Exception {
    Timer timer = new HashedWheelTimer();

    /*专门用来延迟任务*/
    timer.newTimeout(timeout -> {
      info("执行任务1->会休息5s");
      Thread.sleep(5000L);
    }, 1, TimeUnit.SECONDS);

    timer.newTimeout(timeout -> {
      info("执行任务2");
    }, 3, TimeUnit.SECONDS);

  }


  private static void info(String msg) {
    System.out.println(
        "当前时间:" + (FastDateFormat.getInstance("yyyy-MM-dd HH:mm:ss").format(new Date())) + ":打印信息"
            + msg);
  }
}
