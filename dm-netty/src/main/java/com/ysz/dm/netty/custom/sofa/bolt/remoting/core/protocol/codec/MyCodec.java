package com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol.codec;

import io.netty.channel.ChannelHandler;

public interface MyCodec {

  ChannelHandler newEncoder();

  ChannelHandler newDecoder();

}
