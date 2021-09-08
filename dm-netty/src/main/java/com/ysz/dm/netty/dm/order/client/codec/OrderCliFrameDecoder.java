package com.ysz.dm.netty.dm.order.client.codec;

import io.netty.handler.codec.LengthFieldBasedFrameDecoder;

/**
 * @author carl
 */
public class OrderCliFrameDecoder extends LengthFieldBasedFrameDecoder {

  public OrderCliFrameDecoder() {
    super(Integer.MAX_VALUE, 0, 2, 0,
        2);
  }
}
