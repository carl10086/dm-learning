package com.ysz.dm.netty.dm.custom.protocol.codec;

import com.ysz.dm.netty.dm.custom.protocol.command.CustomCommand;
import com.ysz.dm.netty.dm.custom.protocol.constants.CustomCommandType;
import com.ysz.dm.netty.dm.custom.protocol.constants.CustomProtocolCodeType;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CustomCommandEncoderV1Impl implements CustomCommandEncoder {

  private static final Logger logger = LoggerFactory.getLogger(CustomCommandEncoderV1Impl.class);

  @Override
  public void encode(
      final ChannelHandlerContext ctx,
      final CustomCommand msg,
      final ByteBuf out)
      throws Exception {
    int index = out.writerIndex();
    CustomCommandType type = msg.type();

    /*1. protoCode 1 bytes*/
    CustomProtocolCodeType customProtocolCodeType = msg.proto();
    out.writeByte(customProtocolCodeType.getVal());

    /*2. type 1 bytes*/
    out.writeByte(type.getVal());

    /*3. id 4 bytes */
    out.writeInt(msg.id());

    /*4. codec type 1 bytes*/
    out.writeInt(msg.serializeType().getVal());

    /*5. class Name length  2bytes*/
    out.writeShort(msg.classNameLength());

    /*6. header Len 2bytes*/
//    out.writeShort(msg.headerLength());

    /*7. content Len 4bytes*/
    out.writeInt(msg.contentLength());

    /*8. class Name*/
    out.writeBytes(msg.className());

    /*9. Header */
//    out.writeBytes(msg.header());
    /*10. Content*/
    out.writeBytes(msg.content());

  }
}
