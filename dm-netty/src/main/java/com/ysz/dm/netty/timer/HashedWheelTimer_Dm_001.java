package com.ysz.dm.netty.timer;

import io.netty.util.HashedWheelTimer;
import io.netty.util.Timeout;
import io.netty.util.Timer;
import java.util.concurrent.TimeUnit;

public class HashedWheelTimer_Dm_001 {

  public static void main(String[] args) throws Exception {
    Timer timer = new HashedWheelTimer();

    /*专门用来延迟任务*/
    final Timeout timeoutTask = timer.newTimeout(timeout -> {
      System.err.println("5s 后执行任务");
    }, 5, TimeUnit.SECONDS);



  }

}
