package com.ysz.dm.netty.order.server;

import com.ysz.dm.netty.order.domain.MessageReq;
import com.ysz.dm.netty.order.domain.Operation;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;

/**
 * @author carl
 */
public class OrderServerBizHandler extends SimpleChannelInboundHandler<MessageReq> {

  @Override
  protected void channelRead0(ChannelHandlerContext ctx, MessageReq msg) throws Exception {
    Operation operation = msg.getMessageBody();
    ctx.writeAndFlush(operation.execute());
  }
}
