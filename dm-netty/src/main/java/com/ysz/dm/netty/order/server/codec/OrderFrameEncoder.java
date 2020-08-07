package com.ysz.dm.netty.order.server.codec;

import io.netty.handler.codec.LengthFieldPrepender;

/**
 * @author carl
 */
public class OrderFrameEncoder extends LengthFieldPrepender {

  public OrderFrameEncoder() {
    super(2);
  }
}
