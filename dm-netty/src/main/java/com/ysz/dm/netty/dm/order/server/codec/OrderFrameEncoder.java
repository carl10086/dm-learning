package com.ysz.dm.netty.dm.order.server.codec;

import io.netty.handler.codec.LengthFieldPrepender;

/**
 * @author carl
 */
public class OrderFrameEncoder extends LengthFieldPrepender {

  public OrderFrameEncoder() {
    super(2);
  }
}
