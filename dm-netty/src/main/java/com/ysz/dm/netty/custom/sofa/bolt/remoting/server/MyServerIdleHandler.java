package com.ysz.dm.netty.custom.sofa.bolt.remoting.server;

import com.alipay.remoting.util.RemotingUtil;
import io.netty.channel.ChannelDuplexHandler;
import io.netty.channel.ChannelHandler.Sharable;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.timeout.IdleStateEvent;
import lombok.extern.slf4j.Slf4j;

@Sharable
@Slf4j
public class MyServerIdleHandler extends ChannelDuplexHandler {

  @Override
  public void userEventTriggered(final ChannelHandlerContext ctx, final Object evt)
      throws Exception {
    if (evt instanceof IdleStateEvent) {
      /*1. 如果是  Idle 的事件, 关闭下 ctx */
      try {
        log.warn("Connection idle, close it from server side: {}",
            RemotingUtil.parseRemoteAddress(ctx.channel()));
        ctx.close();
      } catch (Exception e) {
        log.warn("Exception caught when closing connection in ServerIdleHandler.", e);
      }
    } else {
      /*2. or else just go on*/
      super.userEventTriggered(ctx, evt);
    }
  }
}