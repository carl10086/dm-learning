package com.ysz.dm.netty.order.domain;

import com.alibaba.fastjson.JSON;
import io.netty.buffer.ByteBuf;
import java.nio.charset.Charset;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
public abstract class Message<T extends MessageBody> {


  protected MessageHeader messageHeader;

  protected T messageBody;


  public abstract Class<? extends T> getMessageBodyDecodeClass(int opCode);

  public void decode(ByteBuf msg) {
    int version = msg.readInt();
    int opCode = msg.readInt();
    long id = msg.readLong();
    this.messageHeader = new MessageHeader(
        version, opCode, id
    );
    this.messageBody = JSON.parseObject(
        msg.toString(Charset.defaultCharset()),
        getMessageBodyDecodeClass(this.messageHeader.getOpCode())
    );
  }

  public void encode(ByteBuf byteBuf) {
    byteBuf.writeInt(messageHeader.getVersion());
    byteBuf.writeInt(messageHeader.getOpCode());
    byteBuf.writeLong(messageHeader.getStreamId());
    byteBuf.writeBytes(JSON.toJSONString(messageBody).getBytes(Charset.defaultCharset()));
  }


}
