package com.ysz.dm.netty.echo;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.*;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioServerSocketChannel;

public class EchoServer {

  public static void main(String[] args) throws Exception {
    /*EventLoopGroup 代表的就是线程池资源*/
    EventLoopGroup boss = new NioEventLoopGroup(1, new ThreadFactoryBuilder()
        .setNameFormat("boss-%d")
        .build());
    EventLoopGroup work = new NioEventLoopGroup(new ThreadFactoryBuilder()
        .setNameFormat("worker-%d")
        .build());
    try {
      ServerBootstrap serverBootstrap = new ServerBootstrap();
      serverBootstrap
          .group(boss, work)
          .channel(NioServerSocketChannel.class)
          .option(ChannelOption.SO_BACKLOG, 5)
          .childHandler(
              new ChannelInitializer<SocketChannel>() {
                @Override
                protected void initChannel(SocketChannel ch) throws Exception {
                  ch.pipeline()
                      .addLast(
                          new ChannelInboundHandlerAdapter() {

                            @Override
                            public void channelRead(ChannelHandlerContext ctx, Object msg)
                                throws Exception {
                              ctx.write(msg);
                            }

                            @Override
                            public void channelReadComplete(ChannelHandlerContext ctx)
                                throws Exception {
                              ctx.flush();
                            }

                            @Override
                            public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause)
                                throws Exception {
                              cause.printStackTrace();
                              ctx.close();
                            }
                          });
                }
              });

      ChannelFuture channelFuture = serverBootstrap.bind(1234).sync();
      // Wait until the server socket is closed.
      channelFuture.channel().closeFuture().sync();
    } catch (Exception e) {
      e.printStackTrace();
    } finally {
      boss.shutdownGracefully();
      work.shutdownGracefully();
    }
  }
}
