package com.ysz.dm.netty.order.client.codec;

import io.netty.handler.codec.LengthFieldPrepender;

/**
 * @author carl
 */
public class OrderCliFrameEncoder extends LengthFieldPrepender {

  public OrderCliFrameEncoder() {
    super(2);
  }
}
