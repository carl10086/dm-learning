package com.ysz.dm.logging.logback;

import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.epoll.Epoll;
import io.netty.channel.epoll.EpollDatagramChannel;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.DatagramChannel;
import io.netty.channel.socket.nio.NioDatagramChannel;
import io.netty.handler.logging.LogLevel;
import io.netty.handler.logging.LoggingHandler;
import java.net.InetSocketAddress;

public class Udp {

  public static void main(String[] args) throws Exception {
    NioEventLoopGroup eventExecutors = new NioEventLoopGroup();
    Bootstrap b = new Bootstrap();
    b.group(eventExecutors)
        .channel(EpollDatagramChannel.class)
        .handler(new ChannelInitializer<NioDatagramChannel>() {
          @Override
          protected void initChannel(NioDatagramChannel channel) throws Exception {
            channel.pipeline().addLast(new UdpStringEncoder(new InetSocketAddress("127.0.0.1", 1234)));
            channel.pipeline().addLast(new LoggingHandler(LogLevel.DEBUG));
          }
        });
    try {
      ChannelFuture sync = b.bind(0).sync();
      Channel channel = sync.channel();
      ChannelFuture aaaaa = channel.writeAndFlush("bbbbbb").sync();
//      ChannelFuture aaaaa = channel.writeAndFlush("cccccc");
      System.out.println(aaaaa.isDone());
      System.out.println(aaaaa.isSuccess());

      channel.closeFuture().await();
    } finally {
      eventExecutors.shutdownGracefully();
    }


  }

}
