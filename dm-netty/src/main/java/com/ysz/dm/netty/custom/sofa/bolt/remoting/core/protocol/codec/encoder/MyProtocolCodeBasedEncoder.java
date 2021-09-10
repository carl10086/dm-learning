package com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol.codec.encoder;

import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.constants.AttributeKeys;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol.MyProtocolCode;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol.MyProtocolManager;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToByteEncoder;
import io.netty.util.Attribute;
import java.io.Serializable;

/**
 * <pre>
 *
 * </pre>
 * @author carl
 */
public class MyProtocolCodeBasedEncoder extends MessageToByteEncoder<Serializable> {

  protected MyProtocolCode protocolCode;

  public MyProtocolCodeBasedEncoder(MyProtocolCode protocolCode) {
    this.protocolCode = protocolCode;
  }

  @Override
  protected void encode(
      final ChannelHandlerContext ctx,
      final Serializable msg,
      final ByteBuf out
  ) throws Exception {
    final MyProtocolCode protocolCode = getProtocolCode(ctx);
    MyProtocolManager.getProtocol(protocolCode).getEncoder().encode(
        ctx, msg, out
    );
  }

  private MyProtocolCode getProtocolCode(ChannelHandlerContext ctx) {
    final Attribute<MyProtocolCode> protocol = ctx.channel().attr(AttributeKeys.PROTOCOL);

    MyProtocolCode protocolCode = null;

    if (protocol != null) {
      protocolCode = protocol.get();
    }

    if (protocolCode == null) {
      protocolCode = this.protocolCode;
    }
    return protocolCode;
  }


}
