package com.ysz.dm.netty.custom.sofa.bolt.remoting.infra.utils;

import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.ServerSocketChannel;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import java.util.concurrent.ThreadFactory;

public class MyNettyEventLoopUtils {

  public static EventLoopGroup newEventLoopGroup(int nThreads, ThreadFactory threadFactory) {
    return
        new NioEventLoopGroup(nThreads, threadFactory);
  }


  public static Class<? extends ServerSocketChannel> getServerSocketChannelClass() {
    return NioServerSocketChannel.class;
  }

}
