package com.ysz.dm.netty.http;

import com.alipay.remoting.util.NettyEventLoopUtil;
import com.ysz.dm.netty.custom.CustomNamedThreadFactory;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.channel.Channel;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.WriteBufferWaterMark;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DtHttpServer {

  private static final Logger LOGGER = LoggerFactory.getLogger(DtHttpServer.class);

  protected volatile ServerBootstrap bootstrap;
  private volatile EventLoopGroup bossGroup;
  private volatile EventLoopGroup workerGroup;

  private DtHttpHandlingSettings settings;

  public static void main(String[] args) throws Exception {
    new DtHttpServer().doStart();
  }


  public void doStart() throws Exception {
    boolean success = false;

    try {
      initEventLoopGroup();
      this.bootstrap = new ServerBootstrap();
      this.bootstrap = new ServerBootstrap();
      this.bootstrap
          .group(bossGroup, workerGroup)
          .channel(NioServerSocketChannel.class)
          .option(ChannelOption.SO_BACKLOG, 1024)
          .option(ChannelOption.SO_REUSEADDR, true)
          .option(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
          .childOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
          .childOption(ChannelOption.TCP_NODELAY, true)
          .childOption(ChannelOption.SO_KEEPALIVE, true)
          .childOption(ChannelOption.WRITE_BUFFER_WATER_MARK,
              new WriteBufferWaterMark(32 * 1024,
                  64 * 1024))
          .childHandler(new DtHttpChannelInitializer())
      ;

      Channel channel = this.bootstrap.bind(1234).sync().channel();
      channel.closeFuture().sync();


    } finally {

      bossGroup.shutdownGracefully();
      workerGroup.shutdownGracefully();

//      if (success == false) {
//        doStop();
//      }

    }


  }


  private void initEventLoopGroup() {
    /*1. work Group 是 daemon 线程*/
    this.workerGroup = NettyEventLoopUtil
        .newEventLoopGroup(
            Runtime
                .getRuntime()
                .availableProcessors() * 2,
            new CustomNamedThreadFactory(
                "Custom-netty-server-worker",
                true));

    /*2. boss Group 是非 daemon 线程, 确保能优雅关机*/
    this.bossGroup = NettyEventLoopUtil
        .newEventLoopGroup(
            1,
            new CustomNamedThreadFactory(
                "Rpc-netty-server-boss",
                false));
  }

  private void doStop() {

  }

}
