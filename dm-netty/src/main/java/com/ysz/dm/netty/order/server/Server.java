package com.ysz.dm.netty.order.server;

import com.ysz.dm.netty.order.Constants;
import com.ysz.dm.netty.order.server.codec.OrderFrameDecoder;
import com.ysz.dm.netty.order.server.codec.OrderFrameEncoder;
import com.ysz.dm.netty.order.server.codec.OrderProtocolDecoder;
import com.ysz.dm.netty.order.server.codec.OrderProtocolEncoder;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.handler.logging.LogLevel;
import io.netty.handler.logging.LoggingHandler;

/**
 * @author carl
 */
public class Server {

  public static void main(String[] args) throws InterruptedException {

    EventLoopGroup boss = new NioEventLoopGroup(1);
    EventLoopGroup worker = new NioEventLoopGroup();
    final OrderServerBizHandler bizHandler = new OrderServerBizHandler();

    try {

      ServerBootstrap b = new ServerBootstrap();
      b.group(boss, worker).channel(NioServerSocketChannel.class)
          .option(ChannelOption.SO_BACKLOG, 100)
          .handler(new LoggingHandler(LogLevel.INFO))
          .childHandler(new ChannelInitializer<SocketChannel>() {
            @Override
            protected void initChannel(SocketChannel ch) throws Exception {
              ChannelPipeline pipeline = ch.pipeline();

              pipeline.addLast(new OrderFrameDecoder());
              pipeline.addLast(new OrderProtocolDecoder());

              pipeline.addLast(new OrderFrameEncoder());
              pipeline.addLast(new OrderProtocolEncoder());
              /*Biz*/
              pipeline.addLast(bizHandler);
            }
          });

      ChannelFuture channelFuture = b.bind(Constants.PORT).sync();
      channelFuture.channel().closeFuture().sync();
    } finally {
      boss.shutdownGracefully();
      worker.shutdownGracefully();
    }


  }

}
