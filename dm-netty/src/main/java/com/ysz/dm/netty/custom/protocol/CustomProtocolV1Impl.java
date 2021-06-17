package com.ysz.dm.netty.custom.protocol;

import com.ysz.dm.netty.custom.protocol.codec.CustomCommandEncoder;
import com.ysz.dm.netty.custom.protocol.codec.CustomCommandEncoderV1Impl;
import com.ysz.dm.netty.custom.protocol.constants.CustomProtocolCodeType;

public class CustomProtocolV1Impl implements CustomProtocol {

  private final CustomCommandEncoder customCommandEncoder = new CustomCommandEncoderV1Impl();


  @Override
  public CustomCommandEncoder encoder() {
    return this.customCommandEncoder;
  }

  @Override
  public CustomProtocolCodeType code() {
    return CustomProtocolCodeType.V1;
  }

  @Override
  public int minDecodeLength() {
    return 1/*protoCode*/ + 1/*command Type*/ + 4/*unique id*/ + 1/*serialize type*/
        + 2/*class Name length*/ + 4/*content Length*/;
  }
}
