package com.ysz.dm.netty.dm.order.server.codec;

import com.ysz.dm.netty.dm.order.domain.MessageReq;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToMessageDecoder;
import java.util.List;

/**
 * @author carl
 */
public class OrderProtocolDecoder extends MessageToMessageDecoder<ByteBuf> {

  @Override
  protected void decode(ChannelHandlerContext ctx, ByteBuf msg, List<Object> out) throws Exception {
    MessageReq messageReq = new MessageReq();
    messageReq.decode(msg);
    out.add(messageReq);
  }
}
