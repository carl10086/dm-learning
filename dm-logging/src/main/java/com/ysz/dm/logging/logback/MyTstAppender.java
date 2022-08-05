package com.ysz.dm.logging.logback;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.UnsynchronizedAppenderBase;

public class MyTstAppender extends UnsynchronizedAppenderBase<ILoggingEvent> {

  String httpUrl;
  String carl;

  @Override
  public void start() {
    super.start();
    System.out.println("start");
  }

  @Override
  protected void append(ILoggingEvent eventObject) {
    System.out.println("接收到了");
  }

  public void setHttpUrl(String httpUrl) {
    this.httpUrl = httpUrl;
  }

  public void setCarl(String carl) {
    this.carl = carl;
  }


  @Override
  public void stop() {
    super.stop();
    System.out.println("stop");
  }
}
