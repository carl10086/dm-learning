package com.ysz.dm.netty.custom.sofa.bolt.remoting.core.constants;

import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol.MyProtocolCode;
import io.netty.util.AttributeKey;

public class AttributeKeys {

  public static final AttributeKey<Integer> HEARTBEAT_COUNT = AttributeKey
      .valueOf("heartbeatCount");

  public static final AttributeKey<Boolean> HEARTBEAT_SWITCH = AttributeKey
      .valueOf("heartbeatSwitch");


  public static final AttributeKey<MyProtocolCode> PROTOCOL = AttributeKey.valueOf("protocol");

  public static final AttributeKey<Byte> VERSION = AttributeKey
      .valueOf("version");

}
