package com.ysz.dm.netty.order.client.codec;

import com.ysz.dm.netty.order.domain.MessageReq;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToMessageEncoder;
import java.util.List;

/**
 * @author carl
 */
public class OrderCliProtocolEncoder extends MessageToMessageEncoder<MessageReq> {

  @Override
  protected void encode(ChannelHandlerContext ctx, MessageReq msg, List<Object> out)
      throws Exception {
    ByteBuf buffer = ctx.alloc().buffer();
    msg.encode(buffer);
    out.add(buffer);
  }
}
