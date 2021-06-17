package com.ysz.dm.netty.custom.protocol.codec;

import com.alipay.remoting.codec.AbstractBatchDecoder;
import com.ysz.dm.netty.custom.protocol.CustomProtocol;
import com.ysz.dm.netty.custom.protocol.CustomProtocolManager;
import com.ysz.dm.netty.custom.protocol.constants.CustomProtocolCodeType;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import java.util.List;

public class CustomCommandDecoder extends AbstractBatchDecoder {


  @Override
  protected void decode(
      final ChannelHandlerContext ctx,
      final ByteBuf in,
      final List<Object> out)
      throws Exception {
    /*1. 先 mark 一下、失败的 时候 reset*/
    in.markReaderIndex();


    /*2. 解码 protocol */
    CustomProtocolCodeType protocolCodeType = decodeProtocolCodeType(in);
    if (null != protocolCodeType) {
      if (in.readableBytes() >= 1) {
        CustomProtocol protocol = CustomProtocolManager.getInstance()
            .findProtocol(protocolCodeType);
        if (null != protocol) {
        }
      }
    }
  }


  private CustomProtocolCodeType decodeProtocolCodeType(final ByteBuf in) {
    if (in.readableBytes() >= 1) {
      return CustomProtocolCodeType.of(in.readByte());
    }
    return null;
  }


}
