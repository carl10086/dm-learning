package com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol.codec;

import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol.MyProtocolCode;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol.codec.encoder.MyProtocolCodeBasedEncoder;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol.version.MyProtocolVersion;
import io.netty.channel.ChannelHandler;

/**
 * <pre>
 *   这一层是 Codec 的 facade
 * </pre>
 */
public class MyRpcCodec implements MyCodec {

  @Override
  public ChannelHandler newEncoder() {
    return new MyProtocolCodeBasedEncoder(
        MyProtocolCode.fromBytes(MyProtocolVersion.V2.getVal())
    );
  }

  @Override
  public ChannelHandler newDecoder() {
    return null;
  }
}
