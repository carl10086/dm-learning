package com.ysz.dm.logging.graylog.netty;

import io.netty.bootstrap.Bootstrap;
import io.netty.buffer.ByteBuf;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.epoll.Epoll;
import io.netty.channel.epoll.EpollDatagramChannel;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.DatagramPacket;
import io.netty.channel.socket.nio.NioDatagramChannel;
import java.io.IOException;
import java.net.InetSocketAddress;

public class NettyUdpClient {

  private NioEventLoopGroup group;

  private Channel channel;

  public NettyUdpClient() {
    this.group = new NioEventLoopGroup(1);

    Bootstrap b = new Bootstrap();

    b.group(group).channel(channel()).handler(init());

    ChannelFuture sync = null;
    try {
      sync = b.bind(0).sync();
      this.channel = sync.channel();
    } catch (Exception e) {
      this.close();
      throw new RuntimeException(e);
    }

    Runtime.getRuntime().addShutdownHook(new Thread(() -> NettyUdpClient.this.close()));
  }


  public void close() {
    if (this.group != null) {
      this.group.shutdownGracefully();
    }
  }

  private ChannelInitializer<? extends Channel> init() {
    return new ChannelInitializer<Channel>() {
      @Override
      protected void initChannel(Channel ch) throws Exception {

      }
    };
  }

  private Class<? extends Channel> channel() {
    if (Epoll.isAvailable()) {
      return EpollDatagramChannel.class;
    }

    return NioDatagramChannel.class;
  }


  public void send(final ByteBuf byteBuf, final InetSocketAddress target) throws IOException {
    this.channel.write(new DatagramPacket(
        byteBuf, target
    ));
  }


  public void flush() {
    this.channel.flush();
  }
}
