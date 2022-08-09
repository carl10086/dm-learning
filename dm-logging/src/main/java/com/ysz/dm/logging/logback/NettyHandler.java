package com.ysz.dm.logging.logback;

import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import java.net.DatagramPacket;

public class NettyHandler extends SimpleChannelInboundHandler<Object> {


  @Override
  public void channelReadComplete(ChannelHandlerContext ctx) throws Exception {
    super.channelReadComplete(ctx);
  }

  @Override
  public void channelRead0(ChannelHandlerContext ctx, Object msg) throws Exception {
    DatagramPacket packet = (DatagramPacket) msg;
    System.out.println("Received Message : ");
  }

  @Override
  public void channelActive(ChannelHandlerContext ctx) throws Exception {
    super.channelActive(ctx);
    System.err.println("channel Active");
  }

}