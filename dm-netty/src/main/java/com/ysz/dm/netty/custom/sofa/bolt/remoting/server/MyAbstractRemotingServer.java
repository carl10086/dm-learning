package com.ysz.dm.netty.custom.sofa.bolt.remoting.server;

import com.google.common.base.Preconditions;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.cfg.MyBoltOption;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.cfg.MyBoltOptionsCfg;
import java.util.concurrent.atomic.AtomicBoolean;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public abstract class MyAbstractRemotingServer implements MyRemotingServer {


  protected String ip;
  protected int port;
  protected MyBoltOptionsCfg options;
  protected final AtomicBoolean isStarted = new AtomicBoolean(false);


  public MyAbstractRemotingServer(String ip, int port) {
    checkPortRange(port);
    this.ip = Preconditions.checkNotNull(ip);
    this.port = port;
    this.options = new MyBoltOptionsCfg(128);
  }

  private void checkPortRange(final int port) {
    Preconditions
        .checkArgument(port > 0 && port <= 65535, "Illegal port! should between 0 and 65535.");
  }

  public <T> MyAbstractRemotingServer updateOption(MyBoltOption<T> option, T value) {
    this.options.updateOption(option, value);
    return this;
  }

  public void startUp() {
    /*这里的抽象没东西阿， 就这 ...*/
    if (isStarted.compareAndSet(false, true)) {
      try {
        doInit();
      } catch (Exception e) {
        log.error("startUp failed", e);
      }
    }
  }

  public <T> T getOption(
      MyBoltOption<T> option
  ) {
    return options.getOption(option);
  }


  protected abstract void doInit();
}
