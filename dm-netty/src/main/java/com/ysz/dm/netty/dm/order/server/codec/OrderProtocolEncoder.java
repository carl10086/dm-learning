package com.ysz.dm.netty.dm.order.server.codec;

import com.ysz.dm.netty.dm.order.domain.MessageResp;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToMessageEncoder;
import java.util.List;

/**
 * @author carl
 */
public class OrderProtocolEncoder extends MessageToMessageEncoder<MessageResp> {

  @Override
  protected void encode(ChannelHandlerContext ctx, MessageResp msg, List<Object> out)
      throws Exception {
    ByteBuf buffer = ctx.alloc().buffer();
    msg.encode(buffer);
    out.add(buffer);
  }
}
