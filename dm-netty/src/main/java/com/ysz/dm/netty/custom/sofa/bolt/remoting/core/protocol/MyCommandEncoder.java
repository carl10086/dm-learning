package com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol;

import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import java.io.Serializable;

public interface MyCommandEncoder {

  /**
   * 对象序列化为 bytes
   * @throws Exception
   */
  void encode(ChannelHandlerContext ctx, Serializable msg, ByteBuf out) throws Exception;
}
