package com.ysz.dm.netty.custom;

import com.alipay.remoting.util.NettyEventLoopUtil;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.WriteBufferWaterMark;
import io.netty.channel.socket.nio.NioServerSocketChannel;

public class CustomNettyServer {

  private ServerBootstrap bootstrap;
  private CustomCfg customCfg;


  private EventLoopGroup bossGroup;
  private EventLoopGroup workerGroup;

  public void doInit() {
    initEventLoopGroup();

    this.bootstrap = new ServerBootstrap();
    this.bootstrap
        .group(bossGroup, workerGroup)
        .channel(NioServerSocketChannel.class)
        .option(ChannelOption.SO_BACKLOG, customCfg.tcp_so_backlog())
        .option(ChannelOption.SO_REUSEADDR, customCfg.tcp_so_reuseaddr())
        .option(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
        .childOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
        .childOption(ChannelOption.TCP_NODELAY, customCfg.tcp_nodelay())
        .childOption(ChannelOption.SO_KEEPALIVE, customCfg.tcp_so_keepalive())
        .childOption(ChannelOption.WRITE_BUFFER_WATER_MARK,
            new WriteBufferWaterMark(customCfg.netty_buffer_low_watermark(),
                customCfg.netty_buffer_high_watermark()))
    ;


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


}
