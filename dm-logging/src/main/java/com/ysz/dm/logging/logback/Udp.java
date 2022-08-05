package com.ysz.dm.logging.logback;

import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioDatagramChannel;

public class Udp {

  public static void main(String[] args) throws Exception {
    Bootstrap b = new Bootstrap();
    b.group(new NioEventLoopGroup()).channel(NioDatagramChannel.class)
        .handler(new ChannelInitializer<NioDatagramChannel>() {
          @Override
          protected void initChannel(NioDatagramChannel channel) throws Exception {
            channel.pipeline().addLast(new NettyHandler());
          }
        });
    ChannelFuture sync = b.connect("127.0.0.1", 1234).sync();
    Channel channel = sync.channel();
    channel.writeAndFlush("aaa");
    channel.close();
  }

}
