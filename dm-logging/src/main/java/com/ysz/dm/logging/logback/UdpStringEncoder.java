package com.ysz.dm.logging.logback;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.socket.DatagramPacket;
import io.netty.handler.codec.MessageToMessageEncoder;
import java.net.InetSocketAddress;
import java.nio.charset.Charset;
import java.util.List;

public class UdpStringEncoder extends MessageToMessageEncoder<String> {

  private final InetSocketAddress remoteAddress;

  public UdpStringEncoder(InetSocketAddress remoteAddress) {
    this.remoteAddress = remoteAddress;
  }

  @Override
  protected void encode(ChannelHandlerContext channelHandlerContext, String str, List<Object> list) throws Exception {
    if (str != null) {
      byte[] bytes = str.getBytes(Charset.defaultCharset());

      ByteBuf byteBuf = Unpooled.copiedBuffer(bytes);
      list.add(new DatagramPacket(byteBuf, remoteAddress));
    }


  }
}
