package com.ysz.dm.netty.custom.protocol.codec;

import com.ysz.dm.netty.custom.protocol.command.CustomCommand;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;

public interface CustomCommandEncoder {


  /**
   * 序列化
   * @param ctx
   * @param msg
   * @param out
   * @throws Exception
   */
  void encode(ChannelHandlerContext ctx, CustomCommand msg, ByteBuf out) throws Exception;
}
