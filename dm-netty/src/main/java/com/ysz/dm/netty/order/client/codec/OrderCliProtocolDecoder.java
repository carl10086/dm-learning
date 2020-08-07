package com.ysz.dm.netty.order.client.codec;

import com.ysz.dm.netty.order.domain.MessageResp;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToMessageDecoder;
import java.util.List;

/**
 * @author carl
 */
public class OrderCliProtocolDecoder extends MessageToMessageDecoder<ByteBuf> {

  @Override
  protected void decode(ChannelHandlerContext ctx, ByteBuf msg, List<Object> out) throws Exception {
    MessageResp resp = new MessageResp();
    resp.decode(msg);
    out.add(resp);
  }
}
