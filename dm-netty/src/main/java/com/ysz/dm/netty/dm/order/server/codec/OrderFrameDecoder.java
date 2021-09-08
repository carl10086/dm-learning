package com.ysz.dm.netty.dm.order.server.codec;

import io.netty.handler.codec.LengthFieldBasedFrameDecoder;

/**
 * @author carl
 */
public class OrderFrameDecoder extends LengthFieldBasedFrameDecoder {

  public OrderFrameDecoder(
  ) {
    super(Integer.MAX_VALUE,
        0,
        2,
        0,
        2);
  }
}
