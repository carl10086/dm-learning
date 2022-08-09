package com.ysz.dm.logging.graylog;

import java.net.InetSocketAddress;
import java.util.concurrent.atomic.AtomicIntegerFieldUpdater;

public class GelfHostsLb {

  private InetSocketAddress[] addresses;

  public GelfHostsLb(final String graylogHosts, final int port) {

    String[] split = graylogHosts.split(",", -1);
    this.addresses = new InetSocketAddress[split.length];

    for (int i = 0; i < split.length; i++) {
      this.addresses[i] = new InetSocketAddress(split[i], port);
    }
  }


  static final AtomicIntegerFieldUpdater<GelfHostsLb> sendUdpCntUpdater = AtomicIntegerFieldUpdater.newUpdater(
      GelfHostsLb.class,
      "sendUdpCnt"
  );

  private volatile int sendUdpCnt = Integer.MAX_VALUE;

  public InetSocketAddress next() {
    return addresses[modulo(sendUdpCntUpdater.incrementAndGet(this), addresses.length)];
  }


  private static int modulo(final int value, final int modulo) {
    return ((value % modulo) + modulo) % modulo;
  }
}
