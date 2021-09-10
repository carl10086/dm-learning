package com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol.codec;

import io.netty.channel.ChannelHandler;

/**
 * <pre>
 *   这一层是 Codec 的 facade
 * </pre>
 */
public class MyRpcCodec implements MyCodec{

  @Override
  public ChannelHandler newEncoder() {
    return null;
  }

  @Override
  public ChannelHandler newDecoder() {
    return null;
  }
}
