package com.ysz.dm.netty.dm.custom.protocol;

import com.ysz.dm.netty.dm.custom.protocol.codec.CustomCommandEncoder;
import com.ysz.dm.netty.dm.custom.protocol.constants.CustomProtocolCodeType;

public interface CustomProtocol {

  default void register() {
    CustomProtocolManager.getInstance().register(this);
  }

  CustomCommandEncoder encoder();

  CustomProtocolCodeType code();

  /**
   * @return 解码最小长度
   */
  int minDecodeLength();
}
